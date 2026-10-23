#include "graphics/host_gpu/renderer/pipeline/pipelineBinaries.h"

#include "common/stringUtils.h"
#include "local-platform.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <system_error>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

constexpr char     Magic[8]   = {'K', 'y', 't', 'y', 'P', 'B', '0', '1'};
constexpr uint64_t FooterSize = 2 * sizeof(uint64_t) + sizeof(Magic);

template <typename T>
void Put(std::string& out, const T& value) {
	out.append(reinterpret_cast<const char*>(&value), sizeof(value));
}

struct Cursor {
	const uint8_t* at;
	const uint8_t* end;
	template <typename T>
	bool Get(T& value) {
		if (static_cast<size_t>(end - at) < sizeof(T)) return false;
		std::memcpy(&value, at, sizeof(T));
		at += sizeof(T);
		return true;
	}
	bool Get(PipelineBinaries::Key& key) {
		return Get(key.size) && key.size <= key.bytes.size() && Get(key.bytes);
	}
};

void PutKey(std::string& out, const PipelineBinaries::Key& key) {
	Put(out, key.size);
	Put(out, key.bytes);
}

PipelineBinaries::Key KeyOf(const VkPipelineBinaryKeyKHR& key) {
	PipelineBinaries::Key out;
	out.size = std::min<uint32_t>(key.keySize, VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR);
	std::memcpy(out.bytes.data(), key.key, out.size);
	return out;
}

} // namespace

size_t PipelineBinaries::KeyHash::operator()(const Key& key) const {
	return XXH3_64bits(key.bytes.data(), key.size);
}

PipelineBinaries::Handles::~Handles() {
	for (const auto binary: owned) {
		if (binary != VK_NULL_HANDLE) VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyPipelineBinaryKHR(device, binary, nullptr);
	}
}

std::string PipelineBinaries::Signature(vk::Device device, const std::string& device_signature) {
	VkPipelineBinaryKeyKHR key {VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
	if (VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPipelineKeyKHR == nullptr ||
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPipelineKeyKHR(device, nullptr, &key) != VK_SUCCESS || key.keySize == 0) {
		return {};
	}
	std::string out = "KytyPipelineBinaries1" + device_signature + ":";
	for (uint32_t i = 0; i < key.keySize && i < VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR; i++) {
		out += fmt::format("{:02x}", key.key[i]);
	}
	return out;
}

bool PipelineBinaries::PipelineKey(vk::Device device, const void* create_info, Key& key) {
	VkPipelineCreateInfoKHR info {VK_STRUCTURE_TYPE_PIPELINE_CREATE_INFO_KHR};
	info.pNext = const_cast<void*>(create_info);
	VkPipelineBinaryKeyKHR out {VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
	if (VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPipelineKeyKHR(device, &info, &out) != VK_SUCCESS || out.keySize == 0) {
		return false;
	}
	key = KeyOf(out);
	return true;
}

std::unique_ptr<PipelineBinaries> PipelineBinaries::Open(vk::Device device, const std::filesystem::path& path,
                                                         const std::string& signature) {
	std::error_code error;
	const auto      size = std::filesystem::file_size(path, error);
	if (signature.empty() || error || size < sizeof(Magic) + sizeof(uint32_t) + FooterSize) return nullptr;
	std::unique_ptr<PipelineBinaries> store(new PipelineBinaries);
	store->m_device = device;
	store->m_file   = LocalPlatform::OpenFileForReading(Common::PathToString(path).c_str());
	const auto read = [&](uint64_t offset, void* data, size_t bytes) {
		return LocalPlatform::ReadScratchFile(store->m_file, offset, data, bytes);
	};
	char        magic[sizeof(Magic)] {};
	uint32_t    signature_size = 0;
	std::string saved;
	if (store->m_file == 0 || !read(0, magic, sizeof(magic)) || std::memcmp(magic, Magic, sizeof(Magic)) != 0 ||
	    !read(sizeof(Magic), &signature_size, sizeof(signature_size)) || signature_size != signature.size()) {
		return nullptr;
	}
	saved.resize(signature_size);
	const uint64_t data_begin = sizeof(Magic) + sizeof(signature_size) + signature_size;
	uint64_t       footer[2] {};
	if (!read(sizeof(Magic) + sizeof(signature_size), saved.data(), saved.size()) || saved != signature ||
	    !read(size - FooterSize, footer, sizeof(footer)) || !read(size - sizeof(Magic), magic, sizeof(magic)) ||
	    std::memcmp(magic, Magic, sizeof(Magic)) != 0 || footer[0] < data_begin || footer[0] > size - FooterSize) {
		return nullptr;
	}
	std::vector<uint8_t> tables(size - FooterSize - footer[0]);
	if (!read(footer[0], tables.data(), tables.size()) || XXH3_64bits(tables.data(), tables.size()) != footer[1]) {
		return nullptr;
	}
	Cursor   cursor {tables.data(), tables.data() + tables.size()};
	uint32_t binaries = 0, pipelines = 0, refs = 0;
	if (!cursor.Get(binaries)) return nullptr;
	store->m_binaries.resize(binaries);
	for (auto& binary: store->m_binaries) {
		if (!cursor.Get(binary.key) || !cursor.Get(binary.offset) || !cursor.Get(binary.size) ||
		    !cursor.Get(binary.hash) || binary.offset < data_begin || binary.offset > footer[0] ||
		    binary.size > footer[0] - binary.offset) {
			return nullptr;
		}
	}
	if (!cursor.Get(pipelines)) return nullptr;
	store->m_pipelines.reserve(pipelines);
	for (uint32_t i = 0; i < pipelines; i++) {
		Key      key;
		Pipeline pipeline;
		if (!cursor.Get(key) || !cursor.Get(pipeline.first) || !cursor.Get(pipeline.count)) return nullptr;
		store->m_pipelines.emplace(key, pipeline);
	}
	if (!cursor.Get(refs)) return nullptr;
	store->m_refs.resize(refs);
	for (auto& ref: store->m_refs) {
		if (!cursor.Get(ref) || ref >= binaries) return nullptr;
	}
	for (const auto& [key, pipeline]: store->m_pipelines) {
		if (pipeline.first > refs || pipeline.count > refs - pipeline.first) return nullptr;
	}
	store->m_users.assign(binaries, 0);
	for (const auto ref: store->m_refs) store->m_users[ref]++;
	store->m_shared.assign(binaries, VK_NULL_HANDLE);
	return store;
}

PipelineBinaries::~PipelineBinaries() {
	for (const auto binary: m_shared) {
		if (binary != VK_NULL_HANDLE) VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyPipelineBinaryKHR(m_device, binary, nullptr);
	}
	LocalPlatform::CloseScratchFile(m_file);
}

uint64_t PipelineBinaries::DataBytes() const {
	uint64_t bytes = 0;
	for (const auto& binary: m_binaries) bytes += binary.size;
	return bytes;
}

bool PipelineBinaries::ReadData(const Binary& binary, std::vector<uint8_t>& data) const {
	data.resize(binary.size);
	return LocalPlatform::ReadScratchFile(m_file, binary.offset, data.data(), data.size()) &&
	       XXH3_64bits(data.data(), data.size()) == binary.hash;
}

bool PipelineBinaries::CreateBinaries(std::span<const uint32_t> indices, VkPipelineBinaryKHR* out) const {
	const auto                           count = static_cast<uint32_t>(indices.size());
	std::vector<std::vector<uint8_t>>    data(count);
	std::vector<VkPipelineBinaryKeyKHR>  keys(count, {VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR});
	std::vector<VkPipelineBinaryDataKHR> datas(count);
	for (uint32_t i = 0; i < count; i++) {
		const auto& binary = m_binaries[indices[i]];
		if (!ReadData(binary, data[i])) return false;
		keys[i].keySize = binary.key.size;
		std::memcpy(keys[i].key, binary.key.bytes.data(), binary.key.size);
		datas[i] = {data[i].size(), data[i].data()};
	}
	VkPipelineBinaryKeysAndDataKHR keys_and_data {count, keys.data(), datas.data()};
	VkPipelineBinaryCreateInfoKHR  create {VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR};
	create.pKeysAndDataInfo = &keys_and_data;
	VkPipelineBinaryHandlesInfoKHR created {VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR};
	created.pipelineBinaryCount = count;
	created.pPipelineBinaries   = out;
	std::fill(out, out + count, VK_NULL_HANDLE);
	if (VULKAN_HPP_DEFAULT_DISPATCHER.vkCreatePipelineBinariesKHR(m_device, &create, nullptr, &created) == VK_SUCCESS &&
	    created.pipelineBinaryCount == count) {
		return true;
	}
	for (uint32_t i = 0; i < count; i++) {
		if (out[i] != VK_NULL_HANDLE) VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyPipelineBinaryKHR(m_device, out[i], nullptr);
		out[i] = VK_NULL_HANDLE;
	}
	return false;
}

PipelineBinaries::Handles PipelineBinaries::Load(const Key& key) const {
	Handles handles;
	handles.device   = m_device;
	const auto found = m_pipelines.find(key);
	if (found == m_pipelines.end() || found->second.count == 0) return handles;
	const auto [first, count] = found->second;
	// Binaries many pipelines share (the driver puts one of 2 MiB into every pipeline) are made once.
	std::vector<VkPipelineBinaryKHR> binaries(count, VK_NULL_HANDLE);
	std::vector<uint32_t>            fresh, fresh_indices;
	for (uint32_t i = 0; i < count; i++) {
		const auto index = m_refs[first + i];
		if (m_users[index] < SharedUsers) {
			fresh.push_back(i);
			fresh_indices.push_back(index);
			continue;
		}
		std::lock_guard lock(m_shared_mutex);
		if (m_shared[index] == VK_NULL_HANDLE && !CreateBinaries({&index, 1}, &m_shared[index])) return handles;
		binaries[i] = m_shared[index];
	}
	handles.owned.resize(fresh.size());
	if (!fresh.empty() && !CreateBinaries(fresh_indices, handles.owned.data())) {
		handles.owned.clear();
		return handles;
	}
	for (size_t j = 0; j < fresh.size(); j++) binaries[fresh[j]] = handles.owned[j];
	handles.binaries = std::move(binaries);
	return handles;
}

// Kept once per key and content: a key alone is not unique (NVIDIA's compression dictionary has one key
// whatever its content).
uint32_t PipelineBinaryWriter::AddBinary(Binary binary) {
	const BinaryId id {binary.key, binary.hash};
	if (const auto found = m_binary_index.find(id); found != m_binary_index.end()) return found->second;
	const auto index = static_cast<uint32_t>(m_binaries.size());
	m_binary_index.emplace(id, index);
	m_binaries.push_back(std::move(binary));
	return index;
}

static bool IsDictionary(const PipelineBinaries::Key& key) {
	static constexpr char tag[] = "_NVDICT_";
	return std::search(key.bytes.begin(), key.bytes.begin() + key.size, tag, tag + sizeof(tag) - 1) !=
	       key.bytes.begin() + key.size;
}

PipelineBinaryWriter::PipelineBinaryWriter(vk::Device device): m_device(device) {
	if (const char* claims = std::getenv("KYTY_PRECOMPILE_CLAIMS"); claims != nullptr && *claims != 0) {
		m_claims = claims;
		std::error_code error;
		std::filesystem::create_directories(m_claims, error);
	}
}

static std::string ClaimName(const PipelineBinaries::Key& key) {
	std::string name;
	for (uint32_t i = 0; i < key.size; i++) name += fmt::format("{:02x}", key.bytes[i]);
	return name;
}

bool PipelineBinaryWriter::Claimed(const PipelineBinaries::Key& key) const {
	std::error_code error;
	if (m_claims.empty() || !std::filesystem::exists(m_claims / ClaimName(key), error)) return false;
	m_claimed.fetch_add(1, std::memory_order_relaxed);
	return true;
}

// After the binaries are held (never before: a shard that is split keeps what it saved, and its halves
// compile the rest).
void PipelineBinaryWriter::Claim(const PipelineBinaries::Key& key) const {
	if (!m_claims.empty()) std::ofstream(m_claims / ClaimName(key), std::ios::binary);
}

bool PipelineBinaryWriter::Capture(const PipelineBinaries::Key& key, vk::Pipeline pipeline, const void* create_info) {
	const auto&                   d = VULKAN_HPP_DEFAULT_DISPATCHER;
	VkPipelineBinaryCreateInfoKHR create {VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR};
	VkPipelineCreateInfoKHR       internal {VK_STRUCTURE_TYPE_PIPELINE_CREATE_INFO_KHR};
	if (create_info != nullptr) {
		internal.pNext             = const_cast<void*>(create_info);
		create.pPipelineCreateInfo = &internal;
	} else {
		create.pipeline = pipeline;
	}
	VkPipelineBinaryHandlesInfoKHR created {VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR};
	PipelineBinaries::Handles      handles;
	handles.device = m_device;
	bool captured  = d.vkCreatePipelineBinariesKHR(m_device, &create, nullptr, &created) == VK_SUCCESS &&
	                created.pipelineBinaryCount != 0;
	if (captured) {
		handles.owned.assign(created.pipelineBinaryCount, VK_NULL_HANDLE);
		created.pPipelineBinaries = handles.owned.data();
		captured = d.vkCreatePipelineBinariesKHR(m_device, &create, nullptr, &created) == VK_SUCCESS;
	}
	if (create_info == nullptr) {
		VkReleaseCapturedPipelineDataInfoKHR release {VK_STRUCTURE_TYPE_RELEASE_CAPTURED_PIPELINE_DATA_INFO_KHR};
		release.pipeline = pipeline;
		(void)d.vkReleaseCapturedPipelineDataKHR(m_device, &release, nullptr);
	}
	if (!captured) return false;
	std::vector<Binary> binaries;
	for (const auto binary: handles.owned) {
		if (binary == VK_NULL_HANDLE) return false;
		VkPipelineBinaryDataInfoKHR info {VK_STRUCTURE_TYPE_PIPELINE_BINARY_DATA_INFO_KHR};
		info.pipelineBinary = binary;
		VkPipelineBinaryKeyKHR binary_key {VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR};
		size_t                 size = 0;
		if (d.vkGetPipelineBinaryDataKHR(m_device, &info, &binary_key, &size, nullptr) != VK_SUCCESS || size == 0) {
			return false;
		}
		Binary out;
		out.data.resize(size);
		if (d.vkGetPipelineBinaryDataKHR(m_device, &info, &binary_key, &size, out.data.data()) != VK_SUCCESS) {
			return false;
		}
		out.data.resize(size);
		out.key  = KeyOf(binary_key);
		out.hash = XXH3_64bits(out.data.data(), out.data.size());
		binaries.push_back(std::move(out));
	}
	if (std::any_of(binaries.begin(), binaries.end(), [](const Binary& binary) { return IsDictionary(binary.key); })) {
		if (m_left_out.fetch_add(1, std::memory_order_relaxed) == 0) {
			std::printf("Pipeline binaries: the driver compresses with its dictionary from here on (after %zu captured, "
			            "%zu created from a store); the rest is left out\n",
			            m_captured.load(), m_copied.load());
		}
		return true;
	}
	std::lock_guard lock(m_mutex);
	auto&           refs = m_pipelines[key];
	refs.clear();
	for (auto& binary: binaries) refs.push_back(AddBinary(std::move(binary)));
	m_captured.fetch_add(1, std::memory_order_relaxed);
	Claim(key);
	return true;
}

void PipelineBinaryWriter::Copy(const PipelineBinaries::Key& key, const PipelineBinaries& from) {
	const auto found = from.m_pipelines.find(key);
	if (found == from.m_pipelines.end()) return;
	m_copied.fetch_add(1, std::memory_order_relaxed);
	std::lock_guard lock(m_mutex);
	auto&           refs = m_pipelines[key];
	refs.clear();
	for (uint32_t i = 0; i < found->second.count; i++) {
		const auto index = from.m_refs[found->second.first + i];
		const auto& binary = from.m_binaries[index];
		refs.push_back(AddBinary({.key = binary.key, .hash = binary.hash, .from = &from, .index = index}));
	}
	Claim(key);
}

void PipelineBinaryWriter::CopyAll(const PipelineBinaries& from) {
	for (const auto& [key, pipeline]: from.m_pipelines) Copy(key, from);
}

size_t PipelineBinaryWriter::Pipelines() const {
	std::lock_guard lock(m_mutex);
	return m_pipelines.size();
}

bool PipelineBinaryWriter::Save(const std::filesystem::path& file_path, const std::string& signature) const {
	std::lock_guard lock(m_mutex);
	std::error_code error;
	std::filesystem::create_directories(file_path.parent_path(), error);
	std::ofstream   file(file_path, std::ios::binary | std::ios::trunc);
	const auto      signature_size = static_cast<uint32_t>(signature.size());
	file.write(Magic, sizeof(Magic));
	file.write(reinterpret_cast<const char*>(&signature_size), sizeof(signature_size));
	file.write(signature.data(), static_cast<std::streamsize>(signature.size()));
	uint64_t    offset = sizeof(Magic) + sizeof(signature_size) + signature.size();
	std::string tables;
	// Only binaries a pipeline uses: one captured or copied again leaves its earlier ones unused.
	std::vector<uint32_t> renumbered(m_binaries.size(), UINT32_MAX);
	std::vector<uint32_t> used;
	for (const auto& [key, list]: m_pipelines) {
		for (const auto ref: list) {
			if (renumbered[ref] != UINT32_MAX) continue;
			renumbered[ref] = static_cast<uint32_t>(used.size());
			used.push_back(ref);
		}
	}
	Put(tables, static_cast<uint32_t>(used.size()));
	std::vector<uint8_t> copied;
	for (const auto ref: used) {
		const auto& binary = m_binaries[ref];
		const auto* data   = &binary.data;
		if (binary.from != nullptr) {
			if (!binary.from->ReadData(binary.from->m_binaries[binary.index], copied)) return false;
			data = &copied;
		}
		file.write(reinterpret_cast<const char*>(data->data()), static_cast<std::streamsize>(data->size()));
		PutKey(tables, binary.key);
		Put(tables, offset);
		Put(tables, static_cast<uint64_t>(data->size()));
		Put(tables, static_cast<uint64_t>(XXH3_64bits(data->data(), data->size())));
		offset += data->size();
	}
	Put(tables, static_cast<uint32_t>(m_pipelines.size()));
	std::vector<uint32_t> refs;
	for (const auto& [key, list]: m_pipelines) {
		PutKey(tables, key);
		Put(tables, static_cast<uint32_t>(refs.size()));
		Put(tables, static_cast<uint32_t>(list.size()));
		for (const auto ref: list) refs.push_back(renumbered[ref]);
	}
	Put(tables, static_cast<uint32_t>(refs.size()));
	for (const auto ref: refs) Put(tables, ref);
	file.write(tables.data(), static_cast<std::streamsize>(tables.size()));
	const uint64_t footer[2] {offset, XXH3_64bits(tables.data(), tables.size())};
	file.write(reinterpret_cast<const char*>(footer), sizeof(footer));
	file.write(Magic, sizeof(Magic));
	return static_cast<bool>(file);
}

} // namespace Libs::Graphics
