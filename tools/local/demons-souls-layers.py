#!/usr/bin/env python3
"""Per-layer render switches for Demon's Souls (PPSA01341 / cp11).

The engine already registers a per-category render control for every layer we
care about; they are simply not bound to anything you can reach from a retail
package build.  This tool binds them: it writes `+convar=value` lines into the
same command-line file the engine reads (`PackageCmdLineArgs.txt`), which is the
only injection surface available (see tools/local/demons-souls-boot-skip.py for
why the emulator cannot pass the guest any arguments).

Every convar below was located by disassembling its registration site in
decrypted/eboot.bin; the object address is the static convar object the engine
fills in, listed so the mapping can be re-checked after a game update.

    layer        convar                                 object      gates
    -----------  -------------------------------------  ----------  ------------------
    particles    showParticle                           0x2f24ea8   all particle systems
    characters   disableDynamicGeometry                 0x2f30f90   dynamic/skinned geometry
    characters   anim_enableSkinning                    0x2f2e988   skeletal skinning
    world        r_disableXPRRendering                  0x2f31cb0   XPR/virtual static world
    vgeo         r_enableVirtualGeometry                0x2f2d508   virtual geometry
    vgeo-cull    r_enableVirtualGeometryOcclusion       0x2f2dae8   virtual geometry occlusion
    fog          fog_enable                             0x2f27c98   volumetric fog
    lighting     r_enableIBL                            0x2f1f668   image based lighting
    ibl-extra    r_enableIBLOcclusion                   0x2f204e8   IBL occlusion
    ibl-extra    r_enableIBLTemporal                    0x2f203f8   IBL temporal accumulation
    shadows      r_enableCascadeShadows                 0x2f1cce0   sun cascade shadow maps
    shadows      r_enableSpotShadows                    0x2f1a190   spot light shadows
    shadows      r_contactShadowsEnabled                0x2f1c300   screen space contact shadows
    ambient      r_enableAmbientShadowing               0x2f1a470   ambient shadowing / AO
    lights       r_enableAllLights                      0x2f19360   every analytic light
    lights-cull  r_enableLightOcclusionCulling          0x2f19900   light occlusion culling
    dof          r_dofEnable                            0x2f0c718   depth of field
    postfx       r_bloomEnable                          0x2f0ba58   bloom
    postfx       filmGrain_enable                       0x2f12f38   film grain
    ui           uiDrawDisable                          0x329b068   HUD/UI drawing
    fur          r_furEnable                            0x2f2c278   fur shells

Everything above is a registered bool convar (constructor 0x450a20) whose name is
referenced exactly once in the image - the registration site.  That single-reference
shape is what a working switch looks like here: lookups are by runtime-computed
name hash, so a consumer never shows up in a cross reference.  Confirming it is the
only static check available; whether a layer costs anything has to be measured.

Usage
-----
    demons-souls-layers.py list
    demons-souls-layers.py apply --disable characters,particles
    demons-souls-layers.py clear
    demons-souls-layers.py status
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BEGIN = '// >>> kyty-layer-switch >>> (managed by tools/local/demons-souls-layers.py)'
END = '// <<< kyty-layer-switch <<<'
ARGS_NAME = 'PackageCmdLineArgs.txt'

# layer -> (human description, convars that switch it off in a shipping build)
LAYERS = {
    'particles': ('特效 / particles', ['+showParticle=false']),
    'characters': ('人物 / dynamic geometry + skinning',
                   ['+disableDynamicGeometry=true', '+anim_enableSkinning=false']),
    'world': ('背景 / static world (XPR virtual geometry)',
              ['+r_disableXPRRendering=true']),
    'vgeo': ('虚拟几何 / virtual geometry', ['+r_enableVirtualGeometry=false']),
    'vgeo-cull': ('虚拟几何遮蔽剔除 / virtual geometry occlusion',
                  ['+r_enableVirtualGeometryOcclusion=false']),
    'fog': ('雾 / volumetric fog', ['+fog_enable=false']),
    'lighting': ('光照 / image based lighting', ['+r_enableIBL=false']),
    'ibl-extra': ('IBL 遮蔽与时域累积 / IBL occlusion + temporal',
                  ['+r_enableIBLOcclusion=false', '+r_enableIBLTemporal=false']),
    'shadows': ('阴影 / cascade + spot + contact shadows',
                ['+r_enableCascadeShadows=false', '+r_enableSpotShadows=false',
                 '+r_contactShadowsEnabled=false']),
    'ambient': ('环境遮蔽 / ambient shadowing', ['+r_enableAmbientShadowing=false']),
    'lights': ('灯光 / every analytic light', ['+r_enableAllLights=false']),
    'lights-cull': ('灯光遮蔽剔除 / light occlusion culling',
                    ['+r_enableLightOcclusionCulling=false']),
    'dof': ('景深 / depth of field', ['+r_dofEnable=false']),
    'postfx': ('后处理 / bloom + film grain',
               ['+r_bloomEnable=false', '+filmGrain_enable=false']),
    'ui': ('界面 / HUD and UI', ['+uiDrawDisable=true']),
    'fur': ('毛发 / fur shells', ['+r_furEnable=false']),
}

# Diagnostic isolates (not "off" switches): render ONLY this, hide everything else.
ISOLATES = {
    'only-particles': ('只画特效 / hide the scene', ['+drawJustParticles=true']),
}


def default_game_dir():
    from_env = os.environ.get('KYTY_GAME', '')
    if from_env and Path(from_env).is_dir():
        return Path(from_env)
    for config in sorted(ROOT.glob('_Build/*/launch-*.json')):
        try:
            command = json.loads(config.read_text())['command']
        except (OSError, ValueError, KeyError):
            continue
        for index, argument in enumerate(command[:-1]):
            if argument == '--game':
                candidate = Path(command[index + 1])
                if candidate.is_dir():
                    return candidate
    return None


def strip_block(text):
    kept, inside = [], False
    for line in text.splitlines():
        if line.strip() == BEGIN:
            inside = True
            continue
        if line.strip() == END:
            inside = False
            continue
        if not inside:
            kept.append(line)
    return '\n'.join(kept).rstrip('\n') + '\n'


def convar_lines(names):
    lines = []
    for name in names:
        if name in LAYERS:
            lines.extend(LAYERS[name][1])
        elif name in ISOLATES:
            lines.extend(ISOLATES[name][1])
        else:
            raise SystemExit(f'unknown layer: {name}')
    return lines


def list_layers(_arguments):
    print('layers (disable with --disable <a,b>):')
    for name, (description, convars) in LAYERS.items():
        print(f'  {name:12} {description:45} {" ".join(convars)}')
    print('isolates (render only this):')
    for name, (description, convars) in ISOLATES.items():
        print(f'  {name:12} {description:45} {" ".join(convars)}')


def apply_layers(arguments):
    names = [n.strip() for n in (arguments.disable or '').split(',') if n.strip()]
    raw = [f'+{item.lstrip("+")}' for item in (arguments.set or [])]
    if not names and not raw:
        raise SystemExit('nothing to apply; pass --disable <layer,...> and/or --set convar=value')
    lines = convar_lines(names) + raw
    names = names + [r[1:] for r in raw]
    args_path = arguments.game / ARGS_NAME
    original = args_path.read_text()
    block = '\n'.join([BEGIN, f'// disabled: {",".join(names)}', *lines, END])
    args_path.write_text(strip_block(original).rstrip('\n') + '\n' + block + '\n')
    print(f'layer switch applied to {args_path}')
    for line in lines:
        print(f'  {line}')


def clear_layers(arguments):
    args_path = arguments.game / ARGS_NAME
    original = args_path.read_text()
    stripped = strip_block(original)
    if stripped == original:
        print('no layer switch installed')
        return
    args_path.write_text(stripped)
    print(f'layer switch cleared from {args_path}')


def show_status(arguments):
    args_path = arguments.game / ARGS_NAME
    text = args_path.read_text() if args_path.is_file() else ''
    print(f'game : {arguments.game}')
    if BEGIN not in text:
        print('layer switch : none')
        return
    inside = False
    print('layer switch : installed')
    for line in text.splitlines():
        if line.strip() == BEGIN:
            inside = True
            continue
        if line.strip() == END:
            inside = False
            continue
        if inside and line.startswith('+'):
            print(f'  {line}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['list', 'apply', 'clear', 'status'])
    parser.add_argument('--game', type=Path, help='game directory (app0)')
    parser.add_argument('--disable', help='comma separated layer names')
    parser.add_argument('--set', action='append', metavar='CONVAR=VALUE',
                        help='raw convar assignment, repeatable (e.g. r_enableSpotShadows=false)')
    arguments = parser.parse_args(argv)

    if arguments.action == 'list':
        return list_layers(arguments)

    if arguments.game is None:
        arguments.game = default_game_dir()
    if arguments.game is None or not arguments.game.is_dir():
        parser.error('could not determine the game directory; pass --game')
    arguments.game = arguments.game.resolve()

    return {'apply': apply_layers, 'clear': clear_layers, 'status': show_status}[arguments.action](arguments)


if __name__ == '__main__':
    sys.exit(main())
