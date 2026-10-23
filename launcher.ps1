param([switch]$DryRun)
# A small settings window in front of run-windows.ps1 and precompile-windows.ps1 (launcher.cmd opens it). The
# settings and the console language names follow the KytyPS5 launcher (src/launcher, by the KytyPS5 developers);
# they are kept in launcher-settings.json next to this script. -DryRun prints the commands without a window.
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$settingsPath = Join-Path $root 'launcher-settings.json'
$languages = 'Japanese', 'English (United States)', 'French (France)', 'Spanish (Spain)', 'German', 'Italian', 'Dutch',
	'Portuguese (Portugal)', 'Russian', 'Korean', 'Chinese (Traditional)', 'Chinese (Simplified)', 'Finnish', 'Swedish',
	'Danish', 'Norwegian', 'Polish', 'Portuguese (Brazil)', 'English (United Kingdom)', 'Turkish', 'Spanish (Latin America)',
	'Arabic', 'French (Canada)', 'Czech', 'Hungarian', 'Greek', 'Romanian', 'Thai', 'Vietnamese', 'Indonesian'
$resolutions = '1280x720', '1920x1080', '2560x1440', '3840x2160'

# The emulator and the precompile program: next to this script in a package (make-package.ps1), else this build tree's.
$buildDir = "$root\_Build\windows-streamline-v20261011"
$emulator = @("$root\kyty_emulator.exe", "$buildDir\kyty_emulator.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
$precompiler = @("$root\kyty_shader_precompile.exe", "$buildDir\kyty_shader_precompile.exe") | Where-Object { Test-Path $_ } |
	Select-Object -First 1

$settings = [ordered]@{ game = ''; resolution = '2560x1440'; fullscreen = $false; aspect = $true; language = 1; redzone = $true;
	ecores = $false; framegen = $false; fpshud = $true; fog = $true;
	spotShadows = $false; contactShadows = $false; sunShadows = $true; depthOfField = $true; bloom = $false; filmGrain = $false;
	ultrawide = $false; renderResolution = 0; vsync = $false; hdr = $false; tab = 'Display' }
if (Test-Path $settingsPath) {
	$saved = Get-Content $settingsPath -Raw | ConvertFrom-Json
	foreach ($property in $saved.PSObject.Properties) { if ($settings.Contains($property.Name)) { $settings[$property.Name] = $property.Value } }
}
if (!$settings.game -and (Test-Path "$root\game-path.txt")) { $settings.game = (Get-Content "$root\game-path.txt" -TotalCount 1).Trim() }

# The efficiency cores (the lowest efficiency class of a hybrid CPU) as an affinity mask, or 0.
function Get-EfficiencyMask {
	Add-Type -Namespace KytyLauncher -Name CpuSets -MemberDefinition @'
[DllImport("kernel32.dll")]
static extern bool GetSystemCpuSetInformation(IntPtr information, uint length, out uint returned, IntPtr process, uint flags);
// Group 0's logical processors as (index, efficiency class) pairs.
public static int[] Query() {
	var result = new System.Collections.Generic.List<int>();
	uint length;
	GetSystemCpuSetInformation(IntPtr.Zero, 0, out length, IntPtr.Zero, 0);
	if (length == 0) return result.ToArray();
	IntPtr buffer = Marshal.AllocHGlobal((int)length);
	try {
		if (!GetSystemCpuSetInformation(buffer, length, out length, IntPtr.Zero, 0)) return result.ToArray();
		for (int offset = 0; offset < length; offset += Marshal.ReadInt32(buffer, offset)) {
			if (Marshal.ReadInt32(buffer, offset + 4) != 0 || Marshal.ReadInt16(buffer, offset + 12) != 0) continue;
			result.Add(Marshal.ReadByte(buffer, offset + 14));
			result.Add(Marshal.ReadByte(buffer, offset + 18));
		}
	} finally {
		Marshal.FreeHGlobal(buffer);
	}
	return result.ToArray();
}
'@
	$values = [KytyLauncher.CpuSets]::Query()
	$sets = for ($i = 0; $i + 1 -lt $values.Count; $i += 2) { [pscustomobject]@{ Cpu = $values[$i]; Class = $values[$i + 1] } }
	if (@($sets | ForEach-Object Class | Sort-Object -Unique).Count -lt 2) { return [int64]0 }
	$slowest = ($sets | Measure-Object Class -Minimum).Minimum
	$mask = [int64]0
	foreach ($set in $sets | Where-Object { $_.Class -eq $slowest -and $_.Cpu -lt 63 }) { $mask = $mask -bor ([int64]1 -shl $set.Cpu) }
	return $mask
}
$efficiencyMask = Get-EfficiencyMask

# The primary screen's size in pixels, whatever the display scaling (the game's fullscreen takes that screen).
function Get-ScreenSize {
	Add-Type -Namespace KytyLauncher -Name Screen -MemberDefinition @'
[DllImport("user32.dll")] static extern IntPtr GetDC(IntPtr window);
[DllImport("user32.dll")] static extern int ReleaseDC(IntPtr window, IntPtr dc);
[DllImport("gdi32.dll")] static extern int GetDeviceCaps(IntPtr dc, int index);
// DESKTOPHORZRES, DESKTOPVERTRES: the physical resolution, not the scaled one.
public static int[] Query() {
	IntPtr dc = GetDC(IntPtr.Zero);
	try { return new int[] { GetDeviceCaps(dc, 118), GetDeviceCaps(dc, 117) }; } finally { ReleaseDC(IntPtr.Zero, dc); }
}
'@
	$size = [KytyLauncher.Screen]::Query()
	if ($size[0] -le 0 -or $size[1] -le 0) { return @(0, 0) }
	return $size
}
$screenSize = Get-ScreenSize
# Ultrawide: a screen of 2:1 or wider (21:9 3440x1440, 2560x1080, 3840x1600; 32:9 5120x1440).
$screenAspect = if ($screenSize[1] -gt 0) { $screenSize[0] / $screenSize[1] } else { 0 }
$ultrawideScreen = $screenAspect -ge 2.0

function Get-PlayCommand {
	$size = $settings.resolution -split 'x'
	$arguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$root\run-windows.ps1`"", '-Prompt', '-Follow',
		'-Exe', "`"$emulator`"",
		'-Width', $size[0], '-Height', $size[1], '-Language', $settings.language)
	# Ultrawide (experimental): the game's render setup takes the screen's aspect ratio instead of 16:9 (a wider view
	# in the same 16:9 image, squeezed), and the fullscreen picture is stretched to the screen, which unsqueezes it
	# (docs\CURRENT-PLAYABLE.md "21:9 / 5K"). The HUD and menus are stretched sideways. The patch: the render setup's
	# "mov dword [rbx+4], 16/9" (eboot.bin 01.007.000) storing width / height of the screen instead. On a screen
	# narrower than 2:1 the option is fullscreen 16:9 with black bars (the stretch would squash the picture).
	# The emulator takes one patch file: the chosen game patches go into game-patch-active.json together (in _Build in
	# a build tree; a package has no _Build: next to this script).
	$mods = @()
	$float = { param([double]$value) [BitConverter]::ToString([BitConverter]::GetBytes([single]$value)).Replace('-', '').ToLower() }
	if ($settings.ultrawide -and $ultrawideScreen) {
		$mods += [pscustomobject]@{ name = "ultrawide: the {0}x{1} screen's aspect ratio {2:N3} in the render setup (was 16:9)" -f $screenSize[0], $screenSize[1], $screenAspect
			enabled = $true; memory = @([pscustomobject]@{ offset = '90ae7f'; off = 'c74304' + (& $float (16 / 9)); on = 'c74304' + (& $float $screenAspect) }) }
		$arguments += '-Fullscreen'
	} elseif ($settings.ultrawide) {
		$arguments += @('-Fullscreen', '-AspectFit')
	} elseif ($settings.fullscreen) { $arguments += '-Fullscreen'; if ($settings.aspect) { $arguments += '-AspectFit' } }
	# Render resolution: the PS5 performance mode's 3D renders lower than 2560x1440 and its 3840x2160 output stays, so
	# the HUD stays sized for 4K and the game's own temporal upscaler fills the output from fewer pixels. Upstream's
	# run-windows.ps1 -RenderHeight does it (src/loader/demonsSoulsRenderSize.cpp: the render setup's constant and the
	# per-mode tables, found in the game's code, so 01.005.000 too).
	$renderHeights = @(0, 1260, 1080, 900)
	$renderHeight = $renderHeights[[Math]::Max(0, [Math]::Min($renderHeights.Count - 1, [int]$settings.renderResolution))]
	if ($renderHeight -gt 0) { $arguments += @('-RenderHeight', $renderHeight) }
	if ($mods.Count -gt 0) {
		$patchDir  = if (Test-Path "$root\_Build") { "$root\_Build" } else { $root }
		$patchFile = "$patchDir\game-patch-active.json"
		$json = [ordered]@{ id = 'PPSA01341'; version = '01.007.000'; process = 'eboot.bin'; mods = @($mods) } | ConvertTo-Json -Depth 6
		[IO.File]::WriteAllText($patchFile, $json, (New-Object System.Text.UTF8Encoding $false))
		$arguments += @('-Patch', "`"$patchFile`"")
	}
	if (!$settings.redzone) { $arguments += '-NoRedZone' }
	# (Upstream v20261014's options.) Presents wait for the display's refresh (Vulkan FIFO): no tearing; adaptive sync
	# displays keep adapting. HDR10 output when Windows HDR is on for the display, SDR otherwise.
	if ($settings.vsync) { $arguments += @('-PresentMode', 'Fifo') }
	if ($settings.hdr) { $arguments += '-Hdr' }
	if ($settings.framegen) { $arguments += @('-FrameGen', '1') }
	$gameConvars = @(
		[pscustomobject]@{ Key = 'fog'; Name = 'fog_enable'; CheckedValue = 'true'; UncheckedValue = 'false' },
		[pscustomobject]@{ Key = 'spotShadows'; Name = 'r_enableSpotShadows'; CheckedValue = 'true'; UncheckedValue = 'false' },
		[pscustomobject]@{ Key = 'contactShadows'; Name = 'r_contactShadowsEnabled'; CheckedValue = 'true'; UncheckedValue = 'false' },
		[pscustomobject]@{ Key = 'sunShadows'; Name = 'r_enableCascadeShadows'; CheckedValue = 'true'; UncheckedValue = 'false' },
		[pscustomobject]@{ Key = 'depthOfField'; Name = 'r_dofEnable'; CheckedValue = 'true'; UncheckedValue = 'false' },
		[pscustomobject]@{ Key = 'bloom'; Name = 'r_bloomEnable'; CheckedValue = 'true'; UncheckedValue = 'false' },
		[pscustomobject]@{ Key = 'filmGrain'; Name = 'filmGrain_enable'; CheckedValue = 'true'; UncheckedValue = 'false' }
	)
	# Always: no XPR index-buffer culling, the game's GPU pass that culls each object's hidden triangles (~2800
	# compute dispatches a frame at Boletaria 1-1, each costing the render thread): the GPU draws them all instead,
	# the image unchanged, ~6% faster (run-windows.ps1).
	$convars = @('doIndexBufferCulling=false'; foreach ($item in $gameConvars) {
		$value = if ($settings[$item.Key]) { $item.CheckedValue } else { $item.UncheckedValue }
		"$($item.Name)=$value"
	})
	$fpsHudValue = if ($settings.fpshud) { '1' } else { '0' }
	# KYTY_TEXTURE_LAYER_RANGE=1: a sampled view of part of a large one-level texture array holds only the layers it
	# reads (upstream v20261014, on by default below 12 GB of video memory): Shrine of Storms' 1.3-1.8 GB texture pool
	# views were each a new whole image (chenxiao07/KytyPS5#9; this build's known issue #1).
	$overrides = "KYTY_FRAMEGEN_RUNTIME_TOGGLE=1,KYTY_FPS_HUD_RUNTIME_TOGGLE=1,KYTY_FPS_HUD=$fpsHudValue,KYTY_TEXTURE_LAYER_RANGE=1,KYTY_GAME_CONVARS=$($convars -join ';')"
	# extra-switches.txt (optional): more KEY=VALUE switches for A/B tests, one per line ('#' starts a comment).
	$extraFile = "$root\extra-switches.txt"
	if (Test-Path $extraFile) {
		$extra = @(Get-Content $extraFile | ForEach-Object { $_.Trim() } | Where-Object { $_ -match '^[A-Za-z_][A-Za-z0-9_]*=' })
		if ($extra) { $overrides += ',' + ($extra -join ',') }
	}
	$arguments += @('-Set', $overrides)
	if ($settings.game) { $arguments += @('-Game', "`"$($settings.game)`"") }
	# The console stays for the live log; after a crash it waits for a key.
	return 'powershell ' + ($arguments -join ' ') + ' & if !errorlevel! neq 0 pause'
}
function Get-PrecompileCommand {
	$arguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$root\precompile-windows.ps1`"",
		'-Exe', "`"$precompiler`"")
	if ($settings.ecores -and $efficiencyMask -ne 0) { $arguments += @('-Affinity', $efficiencyMask) }
	if ($settings.game) { $arguments += @('-Game', "`"$($settings.game)`"") }
	return 'powershell ' + ($arguments -join ' ') + ' & pause'
}
function Save-Settings { $settings | ConvertTo-Json | Set-Content $settingsPath -Encoding UTF8 }
if ($DryRun) {
	"play:       cmd /v:on /s /c `"$(Get-PlayCommand)`""
	"precompile: cmd /s /c `"$(Get-PrecompileCommand)`""
	"efficiency cores: 0x{0:X}" -f $efficiencyMask
	return
}

Add-Type -AssemblyName System.Windows.Forms, System.Drawing
Add-Type -Namespace KytyLauncher -Name Dpi -MemberDefinition '[DllImport("user32.dll")] public static extern bool SetProcessDPIAware();'
[KytyLauncher.Dpi]::SetProcessDPIAware() | Out-Null
[System.Windows.Forms.Application]::EnableVisualStyles()

$form = New-Object System.Windows.Forms.Form -Property @{
	Text = 'KytyPS5 launcher'; FormBorderStyle = 'FixedDialog'; MaximizeBox = $false; StartPosition = 'CenterScreen'
	AutoSize = $true; AutoSizeMode = 'GrowAndShrink'; Font = New-Object System.Drawing.Font('Segoe UI', 9); Padding = '12,12,12,8'
}
# The window's icon (launcher.ico, also the "Demon's Souls" shortcut's), in its title bar and the taskbar.
if (Test-Path "$root\launcher.ico") { try { $form.Icon = New-Object System.Drawing.Icon "$root\launcher.ico" } catch {} }
# The window: the banner, the game folder, then the settings in tabs (Display, Graphics, Performance, System: one
# page shown at a time) and the buttons, which stay below every page.
$grid = New-Object System.Windows.Forms.TableLayoutPanel -Property @{ ColumnCount = 1; AutoSize = $true; Dock = 'Fill' }
$form.Controls.Add($grid)
# The key art across the top (launcher-banner.png; the window works without it).
if (Test-Path "$root\launcher-banner.png") {
	try {
		$banner = New-Object System.Windows.Forms.PictureBox -Property @{ SizeMode = 'Zoom'; Width = 720; Height = 337; Margin = '0,0,0,8'
			Image = [System.Drawing.Image]::FromFile("$root\launcher-banner.png") }
		$grid.Controls.Add($banner)
	} catch {}
}
# A page of rows: a section name (gold, Set-Theme), then the controls side by side.
function New-Page { New-Object System.Windows.Forms.TableLayoutPanel -Property @{ ColumnCount = 2; AutoSize = $true; Margin = '0,0,0,0' } }
function Add-Row($page, $label, [object[]]$controls) {
	$page.Controls.Add((New-Object System.Windows.Forms.Label -Property @{ Text = $label; AutoSize = $true; Anchor = 'Left'; Margin = '0,6,14,6'; Tag = 'section' }))
	$row = New-Object System.Windows.Forms.FlowLayoutPanel -Property @{ AutoSize = $true; WrapContents = $false; Margin = '0,0,0,0' }
	$row.Controls.AddRange($controls)
	$page.Controls.Add($row)
}
$top = New-Page
$grid.Controls.Add($top)
$tabNames = 'Display', 'Graphics', 'Performance', 'System'
$pages = [ordered]@{}
foreach ($name in $tabNames) { $pages[$name] = New-Page }

$game = New-Object System.Windows.Forms.TextBox -Property @{ Text = $settings.game; Width = 440 }
$browse = New-Object System.Windows.Forms.Button -Property @{ Text = 'Browse...'; AutoSize = $true }
$browseZar = New-Object System.Windows.Forms.Button -Property @{ Text = '.zar...'; AutoSize = $true }
Add-Row $top 'Game folder' @($game, $browse, $browseZar)
$title = New-Object System.Windows.Forms.Label -Property @{ AutoSize = $true; ForeColor = 'Gray' }
Add-Row $top '' @($title)
# A game: its folder (eboot.bin and sce_sys) or the folder packed into a ZArchive (a .zar file).
function Read-GameParam([string]$game) {
	if (!$game) { return $null }
	if (Test-Path -LiteralPath "$game\sce_sys\param.json") { return Get-Content -LiteralPath "$game\sce_sys\param.json" -Raw -Encoding UTF8 | ConvertFrom-Json }
	$tool = $precompiler
	if ($game -notmatch '\.zar$' -or !(Test-Path -LiteralPath $game -PathType Leaf) -or !$tool) { return $null }
	$copy = [IO.Path]::GetTempFileName()
	try {
		& $tool --game $game --param $copy 2>$null | Out-Null
		if ($LASTEXITCODE -eq 0) { return Get-Content -LiteralPath $copy -Raw -Encoding UTF8 | ConvertFrom-Json }
	} catch {
	} finally {
		Remove-Item -LiteralPath $copy -ErrorAction SilentlyContinue
	}
	return $null
}
function Update-Title {
	$title.Text = 'Choose the folder with eboot.bin and sce_sys (the game dump), or a .zar archive of it.'
	try {
		$json = Read-GameParam $game.Text.Trim()
		if ($json) {
			$title.Text = '{0} ({1} {2})' -f $json.localizedParameters.($json.localizedParameters.defaultLanguage).titleName,
				$json.titleId, $json.contentVersion
		}
	} catch {}
}
$browse.Add_Click({
	$dialog = New-Object System.Windows.Forms.FolderBrowserDialog -Property @{ Description = 'The game folder (eboot.bin, sce_sys)' }
	if ($game.Text -and (Test-Path -LiteralPath $game.Text -PathType Container)) { $dialog.SelectedPath = $game.Text }
	if ($dialog.ShowDialog($form) -eq 'OK') { $game.Text = $dialog.SelectedPath }
})
# The game folder packed into a ZArchive (zarchive.exe <folder> <file.zar>): read without extracting it.
$browseZar.Add_Click({
	$dialog = New-Object System.Windows.Forms.OpenFileDialog -Property @{ Title = 'The game folder packed into a ZArchive'; Filter = 'ZArchive (*.zar)|*.zar' }
	if ($game.Text -and (Test-Path -LiteralPath $game.Text -PathType Leaf)) { $dialog.InitialDirectory = Split-Path $game.Text }
	if ($dialog.ShowDialog($form) -eq 'OK') { $game.Text = $dialog.FileName }
})
$game.Add_TextChanged({ Update-Title })
Update-Title

# One choice for how the game is shown (the settings keep their keys: ultrawide, fullscreen, aspect).
$ultrawideLabel = "Ultrawide (needs a 21:9 or wider screen; this one is {0}x{1})" -f $screenSize[0], $screenSize[1]
if ($ultrawideScreen) { $ultrawideLabel = "Ultrawide (fills your {0}x{1} screen)" -f $screenSize[0], $screenSize[1] }
$displayModes = $ultrawideLabel, 'Fullscreen, 16:9 with black bars',
	'Fullscreen, stretched to fill the screen', 'Windowed'
$displayMode = New-Object System.Windows.Forms.ComboBox -Property @{ DropDownStyle = 'DropDownList'; Width = 320 }
$displayMode.Items.AddRange($displayModes)
$displayMode.SelectedIndex = if ($settings.ultrawide) { 0 } elseif ($settings.fullscreen -and $settings.aspect) { 1 } elseif ($settings.fullscreen) { 2 } else { 3 }
Add-Row $pages.Display 'Display mode' @($displayMode)
# The window's size (fullscreen takes the screen's).
$resolution = New-Object System.Windows.Forms.ComboBox -Property @{ DropDownStyle = 'DropDownList'; Width = 160 }
$resolution.Items.AddRange($resolutions)
$resolution.SelectedItem = if ($resolutions -contains $settings.resolution) { $settings.resolution } else { '2560x1440' }
Add-Row $pages.Display 'Window size' @($resolution)
$resolution.Enabled = $displayMode.SelectedIndex -eq 3
$displayMode.Add_SelectedIndexChanged({ $resolution.Enabled = $displayMode.SelectedIndex -eq 3 })
$renderResolutions = 'Native (2560x1440)', '1260p (2240x1260)', '1080p (1920x1080)', '900p (1600x900)'
$renderResolution = New-Object System.Windows.Forms.ComboBox -Property @{ DropDownStyle = 'DropDownList'; Width = 320 }
$renderResolution.Items.AddRange($renderResolutions)
$renderResolution.SelectedIndex = [Math]::Max(0, [Math]::Min($renderResolutions.Count - 1, [int]$settings.renderResolution))
Add-Row $pages.Display 'Render resolution' @($renderResolution)
$hdr = New-Object System.Windows.Forms.CheckBox -Property @{ Text = 'HDR10 output (turn on "Use HDR" in Windows display settings first)'; AutoSize = $true; Checked = [bool]$settings.hdr }
Add-Row $pages.Display 'HDR' @($hdr)
$vsync = New-Object System.Windows.Forms.CheckBox -Property @{ Text = 'No screen tearing (with Frame Generation, G-SYNC is the better choice)'; AutoSize = $true; Checked = [bool]$settings.vsync }
Add-Row $pages.Display 'VSync' @($vsync)
$language = New-Object System.Windows.Forms.ComboBox -Property @{ DropDownStyle = 'DropDownList'; Width = 320 }
$language.Items.AddRange($languages)
$language.SelectedIndex = [Math]::Max(0, [Math]::Min($languages.Count - 1, [int]$settings.language))
Add-Row $pages.System 'Console language' @($language)
$gameEffects = @(
	[pscustomobject]@{ Key = 'fog'; Label = 'Volumetric fog (about +4.6% FPS when off)' },
	[pscustomobject]@{ Key = 'spotShadows'; Label = 'Torch and spot-light shadows' },
	[pscustomobject]@{ Key = 'contactShadows'; Label = 'Contact shadows (with torch shadows about +3% when off)' },
	[pscustomobject]@{ Key = 'sunShadows'; Label = 'Sun shadows (about +8.5% when off, looks flatter)' },
	[pscustomobject]@{ Key = 'depthOfField'; Label = 'Depth of field' },
	[pscustomobject]@{ Key = 'bloom'; Label = 'Bloom' },
	[pscustomobject]@{ Key = 'filmGrain'; Label = 'Film grain' }
)
$effectCheckboxes = @{}
foreach ($effect in $gameEffects) {
	$checkbox = New-Object System.Windows.Forms.CheckBox -Property @{ Text = $effect.Label; AutoSize = $true; Checked = [bool]$settings[$effect.Key] }
	$effectCheckboxes[$effect.Key] = $checkbox
	Add-Row $pages.Graphics $(if ($effect -eq $gameEffects[0]) { 'Game effects' } else { '' }) @($checkbox)
}
$effectsNote = New-Object System.Windows.Forms.Label -Property @{ Text = 'Unticked effects are off in the game: more FPS, a plainer picture.'; AutoSize = $true; ForeColor = 'Gray'; Margin = '3,6,0,0' }
Add-Row $pages.Graphics '' @($effectsNote)
$framegen = New-Object System.Windows.Forms.CheckBox -Property @{ Text = 'DLSS Frame Generation (RTX 40/50 series; F2 toggles in game)'; AutoSize = $true; Checked = [bool]$settings.framegen }
Add-Row $pages.Performance 'Frame Generation' @($framegen)
$fpshud = New-Object System.Windows.Forms.CheckBox -Property @{ Text = 'Show the FPS counter (F1 toggles in game)'; AutoSize = $true; Checked = [bool]$settings.fpshud }
Add-Row $pages.Performance 'FPS counter' @($fpshud)
$toggleNote = New-Object System.Windows.Forms.Label -Property @{ Text = 'In game: F1 toggles the FPS counter; F2 toggles Frame Generation.'; AutoSize = $true; ForeColor = 'Gray'; Margin = '3,6,0,0' }
Add-Row $pages.Performance '' @($toggleNote)
$redzone = New-Object System.Windows.Forms.CheckBox -Property @{ Text = 'Red-zone protection (recommended)'; AutoSize = $true; Checked = [bool]$settings.redzone }
Add-Row $pages.System 'Stability' @($redzone)
$ecores = New-Object System.Windows.Forms.CheckBox -Property @{ Text = 'Run "Precompile shaders" on the efficiency cores only (the PC stays responsive)'; AutoSize = $true
	Checked = ([bool]$settings.ecores -and $efficiencyMask -ne 0); Enabled = ($efficiencyMask -ne 0) }
Add-Row $pages.System 'Precompile' @($ecores)

# The tab bar (themed buttons: the system tab control cannot be drawn dark) over the pages, a line under it.
$nav = New-Object System.Windows.Forms.FlowLayoutPanel -Property @{ AutoSize = $true; WrapContents = $false; Margin = '0,12,0,0' }
$tabButtons = @{}
foreach ($name in $tabNames) {
	$tab = New-Object System.Windows.Forms.Button -Property @{ Text = $name; AutoSize = $true; Margin = '0,0,2,0'; Tag = 'tab' }
	$tab.Add_Click({ Show-Tab $this.Text })
	$tabButtons[$name] = $tab
	[void]$nav.Controls.Add($tab)
}
$grid.Controls.Add($nav)
$rule = New-Object System.Windows.Forms.Panel -Property @{ Height = 1; Width = 720; Margin = '0,0,0,8' }
$grid.Controls.Add($rule)
$content = New-Object System.Windows.Forms.Panel -Property @{ AutoSize = $false; Margin = '0,0,0,0' }
foreach ($page in $pages.Values) { $page.Location = New-Object System.Drawing.Point(0, 0); $content.Controls.Add($page) }
$grid.Controls.Add($content)
$script:currentTab = if ($tabNames -contains $settings.tab) { $settings.tab } else { 'Display' }
function Show-Tab([string]$name) {
	foreach ($key in $pages.Keys) {
		$pages[$key].Visible = $key -eq $name
		$tabButtons[$key].ForeColor = if ($key -eq $name) { $theme.Gold } else { $theme.Muted }
		$tabButtons[$key].BackColor = if ($key -eq $name) { $theme.Hover } else { $theme.Back }
	}
	$script:currentTab = $name
}

$buttons = New-Object System.Windows.Forms.FlowLayoutPanel -Property @{ AutoSize = $true; Margin = '0,12,0,0' }
$play = New-Object System.Windows.Forms.Button -Property @{ Text = 'Play'; AutoSize = $true; Font = New-Object System.Drawing.Font('Segoe UI', 9, [System.Drawing.FontStyle]::Bold) }
$precompile = New-Object System.Windows.Forms.Button -Property @{ Text = 'Precompile shaders'; AutoSize = $true }
$logs = New-Object System.Windows.Forms.Button -Property @{ Text = 'Logs folder'; AutoSize = $true }
$readme = New-Object System.Windows.Forms.Button -Property @{ Text = 'README'; AutoSize = $true }
$buttons.Controls.AddRange(@($play, $precompile, $logs, $readme))
$grid.Controls.Add($buttons)
$credit = New-Object System.Windows.Forms.Label -Property @{ AutoSize = $true; ForeColor = 'Gray'; Margin = '0,8,0,0'
	Text = 'Settings follow the KytyPS5 launcher (src/launcher) by the KytyPS5 developers.' }
$grid.Controls.Add($credit)
$form.AcceptButton = $play

function Read-Form {
	$settings.game       = $game.Text.Trim()
	$settings.resolution = [string]$resolution.SelectedItem
	$settings.ultrawide  = $displayMode.SelectedIndex -eq 0
	$settings.fullscreen = $displayMode.SelectedIndex -in 1, 2
	$settings.aspect     = $displayMode.SelectedIndex -eq 1
	$settings.renderResolution = $renderResolution.SelectedIndex
	$settings.language   = $language.SelectedIndex
	$settings.redzone    = $redzone.Checked
	$settings.vsync      = $vsync.Checked
	$settings.hdr        = $hdr.Checked
	$settings.framegen   = $framegen.Checked
	$settings.fpshud     = $fpshud.Checked
	$settings.ecores     = $ecores.Checked
	$settings.tab        = $script:currentTab
	foreach ($effect in $gameEffects) { $settings[$effect.Key] = $effectCheckboxes[$effect.Key].Checked }
	Save-Settings
}
$play.Add_Click({
	Read-Form
	Start-Process cmd -ArgumentList "/v:on /s /c `"$(Get-PlayCommand)`"" -WorkingDirectory $root
	$form.Close()
})
$precompile.Add_Click({
	Read-Form
	Start-Process cmd -ArgumentList "/s /c `"$(Get-PrecompileCommand)`"" -WorkingDirectory $root
})
$logs.Add_Click({
	$folder = if (Test-Path "$root\_Build") { "$root\_Build\run-logs" } else { "$root\logs" }
	New-Item -ItemType Directory -Force $folder | Out-Null
	Start-Process explorer.exe $folder
})
$readme.Add_Click({
	$file = @("$root\README.md", "$root\docs\PORTABLE-README.md") | Where-Object { Test-Path $_ } | Select-Object -First 1
	if ($file) { Start-Process notepad.exe "`"$file`"" }
})
$form.Add_FormClosing({ Read-Form })

# Demon's Souls look: near-black like the banner's fade, bone-white text, gold serif section names, dark inputs, a
# blood-red Play button.
$theme = @{
	Back   = [System.Drawing.Color]::FromArgb(18, 17, 16)
	Input  = [System.Drawing.Color]::FromArgb(31, 28, 25)
	Button = [System.Drawing.Color]::FromArgb(42, 37, 32)
	Hover  = [System.Drawing.Color]::FromArgb(58, 49, 40)
	Border = [System.Drawing.Color]::FromArgb(107, 90, 58)
	Text   = [System.Drawing.Color]::FromArgb(226, 220, 207)
	Gold   = [System.Drawing.Color]::FromArgb(201, 165, 90)
	Muted  = [System.Drawing.Color]::FromArgb(140, 133, 122)
	Blood  = [System.Drawing.Color]::FromArgb(122, 31, 26)
	BloodHover = [System.Drawing.Color]::FromArgb(150, 40, 33)
}
$serif = New-Object System.Drawing.Font('Palatino Linotype', 10, [System.Drawing.FontStyle]::Bold)
function Set-Theme($control) {
	foreach ($child in $control.Controls) {
		switch ($child.GetType().Name) {
			'Label' {
				if ($child.ForeColor.Name -eq 'Gray') { $child.ForeColor = $theme.Muted }
				elseif ($child.Tag -eq 'section' -and $child.Text) { $child.ForeColor = $theme.Gold; $child.Font = $serif }
				else { $child.ForeColor = $theme.Text }
			}
			'CheckBox' { $child.ForeColor = $theme.Text }
			'TextBox' { $child.BackColor = $theme.Input; $child.ForeColor = $theme.Text; $child.BorderStyle = 'FixedSingle' }
			'ComboBox' {
				# A drop-down list keeps the system's white box unless it draws its items itself.
				$child.FlatStyle = 'Flat'; $child.BackColor = $theme.Input; $child.ForeColor = $theme.Text
				$child.DrawMode  = 'OwnerDrawFixed'
				$child.Add_DrawItem({
					param($sender, $e)
					$selected = ($e.State -band [System.Windows.Forms.DrawItemState]::Selected) -ne 0
					$back = if ($selected) { $theme.Hover } else { $theme.Input }
					$fore = if ($sender.Enabled) { $theme.Text } else { $theme.Muted }
					$brush = New-Object System.Drawing.SolidBrush $back
					$e.Graphics.FillRectangle($brush, $e.Bounds)
					$brush.Dispose()
					if ($e.Index -ge 0) {
						[System.Windows.Forms.TextRenderer]::DrawText($e.Graphics, [string]$sender.Items[$e.Index], $sender.Font, $e.Bounds, $fore,
							[System.Windows.Forms.TextFormatFlags]::VerticalCenter -bor [System.Windows.Forms.TextFormatFlags]::Left)
					}
				})
			}
			'Button' {
				$child.FlatStyle = 'Flat'; $child.BackColor = $theme.Button; $child.ForeColor = $theme.Text
				$child.FlatAppearance.BorderColor = $theme.Border; $child.FlatAppearance.MouseOverBackColor = $theme.Hover
				$child.Padding = '8,2,8,2'
			}
		}
		Set-Theme $child
	}
}
$form.BackColor = $theme.Back
$form.ForeColor = $theme.Text
$form.Text      = "Demon's Souls - KytyPS5"
Set-Theme $form
$play.Text = 'PLAY'
$play.Font = New-Object System.Drawing.Font('Palatino Linotype', 11, [System.Drawing.FontStyle]::Bold)
$play.BackColor = $theme.Blood
$play.FlatAppearance.BorderColor = $theme.Gold
$play.FlatAppearance.MouseOverBackColor = $theme.BloodHover
$play.Padding = '18,2,18,2'
# The tabs: flat, no border, the chosen one gold on a lighter band (Show-Tab); a thin gold-brown line under the bar.
$tabFont = New-Object System.Drawing.Font('Segoe UI Semibold', 10)
foreach ($tab in $tabButtons.Values) {
	$tab.FlatAppearance.BorderSize = 0; $tab.FlatAppearance.MouseOverBackColor = $theme.Hover
	$tab.Font = $tabFont; $tab.Padding = '14,4,14,4'
}
$rule.BackColor = $theme.Border
# The page area: as large as the largest page, so switching tabs never resizes the window.
$pageWidth = 720; $pageHeight = 0
foreach ($page in $pages.Values) {
	$size = $page.GetPreferredSize([System.Drawing.Size]::Empty)
	$pageWidth = [Math]::Max($pageWidth, $size.Width); $pageHeight = [Math]::Max($pageHeight, $size.Height)
}
$content.Size = New-Object System.Drawing.Size($pageWidth, $pageHeight)
Show-Tab $script:currentTab
$form.ActiveControl = $play # (not the game folder box: its text opened selected)
# A dark title bar (Windows 10 2004+ / 11; elsewhere the call does nothing).
$form.Add_HandleCreated({
	try {
		Add-Type -Namespace KytyLauncher -Name Dwm -MemberDefinition '[DllImport("dwmapi.dll")] public static extern int DwmSetWindowAttribute(IntPtr hwnd, int attribute, ref int value, int size);' -ErrorAction SilentlyContinue
		$dark = 1
		[void][KytyLauncher.Dwm]::DwmSetWindowAttribute($form.Handle, 20, [ref]$dark, 4)
	} catch {}
})
[void]$form.ShowDialog()
