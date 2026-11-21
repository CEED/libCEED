// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Internal header for the caches of the NVRTC and hipRTC JiT compilations of the CUDA and HIP backends
#pragma once

#include <ceed/backend.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

//------------------------------------------------------------------------------
// Caches of JiT compiled binaries
//
// Applications often JiT compile the same kernel source many times, e.g. when they rebuild operators with new data, and
// again in every run. Successful compilations are kept:
// - in memory, for the lifetime of the process, keyed by the compiler options and the source. The files that the source
//   includes, such as the QFunction source, are assumed not to change while the process runs.
// - on disk, in the cache directory, for later processes. NVRTC and hipRTC cannot preprocess a source without compiling
//   it, so the key also has the compiler and the path and contents of every file that the source includes, found by
//   scanning the include directives. A source which may include a file in a way the scan does not follow is only
//   cached in memory. The cache directory and its entries must belong to the user and not be writable by others.
// Failures to use the disk are not errors: they only cause compilations.
//------------------------------------------------------------------------------

/// A JiT compilation, looked up with @ref CeedJitCacheFind() and stored with @ref CeedJitCacheStore()
struct CeedJitCacheEntry {
  std::string                                      key;     // compiler options and source
  std::shared_ptr<const std::string>               binary;  // cached binary, if found
  std::string                                      dir;     // disk cache directory
  std::string                                      hash;    // hash of the key on disk
  std::string                                      path;    // entry file, or empty if not cached on disk
  std::vector<std::pair<std::string, struct stat>> files;   // included files, to check that they do not change
  bool                                             is_found_on_disk = false;
};

// In internal namespace to avoid polluting files that include this
namespace ceed {
namespace internal {

// Format of the first line of an entry file, followed by the key hash and the binary hash
static const char jit_cache_format[] = "libCEED JiT cache 1";

// Largest file read by the caches
static const off_t jit_cache_max_file_size = 256 * 1024 * 1024;

// An include directive, `#include "name"` (quoted) or `#include <name>`
struct JitInclude {
  bool        is_quoted;
  std::string name;
};

// An included file, read again only if it changes
struct JitIncludedFile {
  struct stat             file_stat;
  std::string             hash;           // of the contents
  bool                    is_followable;  // whether the scan follows all its includes
  std::vector<JitInclude> includes;
};

// A disk cache directory, checked once per process
struct JitCacheDir {
  bool is_private;   // belongs to the user and is not writable by others
  bool is_writable;  // false once a write failed because the directory is read-only or full
};

// An included file and the directory it was found in, as (device, inode) pairs. The compiler finds the quoted includes
// of a file from the directory it found the file in, so a file found in two directories is followed for each.
typedef std::tuple<dev_t, ino_t, dev_t, ino_t> JitFileInDir;

// State of the caches, shared by all Ceed contexts
struct JitCacheState {
  std::mutex                                                          mutex;
  std::unordered_map<std::string, std::shared_ptr<const std::string>> binaries;  // by key
  std::unordered_map<std::string, JitIncludedFile>                    files;     // by path
  std::unordered_map<std::string, JitCacheDir>                        dirs;      // by path
};

// The state is never destroyed, so that it outlives any use at exit
static inline JitCacheState &GetJitCacheState() {
  static JitCacheState *state = new JitCacheState;
  return *state;
}

// 64-bit FNV-1a hash, in hexadecimal; see http://www.isthe.com/chongo/tech/comp/fnv/
//
// This is not a cryptographic hash: the cache directory is private to the user, so only accidental collisions matter,
// which 64 bits make negligible for any realistic number of entries (the Clang path uses std::hash similarly).
static inline std::string JitCacheHash(const char *data, size_t size) {
  const uint64_t fnv_offset_basis = 14695981039346656037ull;
  const uint64_t fnv_prime        = 1099511628211ull;
  uint64_t       hash             = fnv_offset_basis;
  char           hex[17];

  for (size_t i = 0; i < size; i++) {
    hash ^= static_cast<unsigned char>(data[i]);
    hash *= fnv_prime;
  }
  snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(hash));
  return hex;
}

// Whether two `stat()` results are of the same, unchanged file
static inline bool JitCacheIsSameFile(const struct stat &a, const struct stat &b) {
  return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_size == b.st_size && a.st_mtim.tv_sec == b.st_mtim.tv_sec &&
         a.st_mtim.tv_nsec == b.st_mtim.tv_nsec && a.st_ctim.tv_sec == b.st_ctim.tv_sec && a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
}

// Whether a file belongs to the user and is not writable by others
static inline bool JitCacheIsPrivate(const struct stat &file_stat) {
  return file_stat.st_uid == geteuid() && !(file_stat.st_mode & (S_IWGRP | S_IWOTH));
}

// Whether `text` has, from `pos`, a `#`, `%:` (digraph), or `??=` (trigraph) followed by a word of a directive that may
// include a file
static inline bool JitCacheMayInclude(const std::string &text, size_t pos) {
  const size_t marker = std::min({text.find('#', pos), text.find("%:", pos), text.find("?\?=", pos)});

  if (marker == std::string::npos) return false;
  for (const char *word : {"include", "import", "embed"}) {
    if (text.find(word, marker) != std::string::npos) return true;
  }
  return false;
}

/**
  @brief Read a regular file of limited size, which must also be private and not a symbolic link if it is an entry file

  @param[in]  ceed      `Ceed` context
  @param[in]  path      Path of the file
  @param[in]  is_entry  Whether the file is an entry file of the disk cache
  @param[out] contents  Contents of the file
  @param[out] file_stat `stat()` result of the file
  @param[out] is_read   Whether the file was read

  @return An error code: 0 - success, otherwise - failure
**/
static inline int JitCacheReadFile(Ceed ceed, const std::string &path, bool is_entry, std::string *contents, struct stat *file_stat, bool *is_read) {
  // O_NONBLOCK: do not wait if the path is a FIFO
  const int fd = open(path.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC | (is_entry ? O_NOFOLLOW : 0));

  *is_read = false;
  if (fd < 0) return CEED_ERROR_SUCCESS;
  if (fstat(fd, file_stat) == 0 && S_ISREG(file_stat->st_mode) && file_stat->st_size <= jit_cache_max_file_size &&
      (!is_entry || JitCacheIsPrivate(*file_stat))) {
    size_t num_read = 0;

    contents->resize(static_cast<size_t>(file_stat->st_size));
    while (num_read < contents->size()) {
      const ssize_t n = read(fd, &(*contents)[num_read], contents->size() - num_read);

      if (n <= 0) break;
      num_read += static_cast<size_t>(n);
    }
    *is_read = num_read == contents->size();
  }
  close(fd);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Find the include directives of a source

  The source is not followable if it may include a file in another way, which the scan does not follow: e.g. with a macro, `#include_next`, or `__has_include`, after a comment, or spelled with a digraph or trigraph.
  It is not followable either if NVRTC and hipRTC may read it differently: unlike NVRTC, hipRTC (Clang) accepts a UTF-8 byte order mark, lone carriage returns, and spaces between a backslash and a new line.

  @param[in]  ceed          `Ceed` context
  @param[in]  source        Source to scan
  @param[out] includes      Include directives of the source
  @param[out] is_followable Whether all the ways the source may include a file are in `includes`

  @return An error code: 0 - success, otherwise - failure
**/
static inline int JitCacheParseIncludes(Ceed ceed, const std::string &source, std::vector<JitInclude> *includes, bool *is_followable) {
  // `#include "name"` or `#include <name>`, at the start of a line
  static const std::regex include_directive(R"regex(^[ \t]*#[ \t]*include[ \t]*(?:"([^"]+)"|<([^>]+)>))regex");
  std::string             text;

  *is_followable = false;
  if (source.compare(0, 3, "\xEF\xBB\xBF") == 0 || source.find("?\?/") != std::string::npos || source.find("__has_include") != std::string::npos ||
      source.find("__has_embed") != std::string::npos) {
    return CEED_ERROR_SUCCESS;
  }
  for (size_t i = 0; i < source.size(); i++) {
    const bool is_lone_carriage_return = source[i] == '\r' && source.compare(i + 1, 1, "\n") != 0;
    bool       is_backslash_spaces_end = false;

    if (source[i] == '\\') {
      const size_t next = source.find_first_not_of(" \t\v\f", i + 1);

      is_backslash_spaces_end = next != i + 1 && next != std::string::npos && (source[next] == '\n' || source[next] == '\r');
    }
    if (is_lone_carriage_return || is_backslash_spaces_end) return CEED_ERROR_SUCCESS;
  }

  // Join the lines continued with a backslash
  for (size_t i = 0; i < source.size(); i++) {
    if (source.compare(i, 2, "\\\n") == 0) {
      i++;
    } else if (source.compare(i, 3, "\\\r\n") == 0) {
      i += 2;
    } else {
      text += source[i];
    }
  }

  std::istringstream lines(text);

  for (std::string line; std::getline(lines, line);) {
    std::smatch  match;
    const size_t first = line.find_first_not_of(" \t\v\f\r");
    const bool   is_directive =
        first != std::string::npos && (line[first] == '#' || line.compare(first, 2, "%:") == 0 || line.compare(first, 3, "?\?=") == 0);

    if (std::regex_search(line, match, include_directive)) {
      // The rest of the line must not include another file, e.g. after the end of a comment
      if (JitCacheMayInclude(match.suffix().str(), 0)) return CEED_ERROR_SUCCESS;
      includes->push_back({match[1].matched, match[1].matched ? match[1].str() : match[2].str()});
    } else if (JitCacheMayInclude(line, is_directive ? first : line.find("*/"))) {
      // Any other directive that may include a file, or a directive after the end of a comment
      return CEED_ERROR_SUCCESS;
    }
  }
  *is_followable = true;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get an included file, reading it again only if it changed

  @param[in]  ceed      `Ceed` context
  @param[in]  path      Path of the file
  @param[in]  file_stat `stat()` result of the file
  @param[out] file      Included file

  @return An error code: 0 - success, otherwise - failure
**/
static inline int JitCacheGetFile(Ceed ceed, const std::string &path, const struct stat &file_stat, JitIncludedFile *file) {
  JitCacheState &state = GetJitCacheState();
  std::string    contents;
  bool           is_read;

  {
    std::lock_guard<std::mutex> lock(state.mutex);
    const auto                  cached = state.files.find(path);

    if (cached != state.files.end() && JitCacheIsSameFile(cached->second.file_stat, file_stat)) {
      *file = cached->second;
      return CEED_ERROR_SUCCESS;
    }
  }
  file->file_stat     = file_stat;
  file->is_followable = false;
  CeedCallBackend(JitCacheReadFile(ceed, path, false, &contents, &file->file_stat, &is_read));
  if (is_read) CeedCallBackend(JitCacheParseIncludes(ceed, contents, &file->includes, &file->is_followable));
  file->hash = JitCacheHash(contents.data(), contents.size());
  {
    std::lock_guard<std::mutex> lock(state.mutex);

    state.files[path] = *file;
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Add the path and contents hash of every file that include directives include, recursively, to a disk cache key

  Like the compiler, quoted headers are found from the directory of the including file and from the include directories, others from the include directories.
  Every file found is followed, so the key covers whichever one the compiler uses.

  @param[in]     ceed          `Ceed` context
  @param[in]     includes      Include directives of a file
  @param[in]     dir           Directory of the file, empty for the source, i.e. the current directory
  @param[in]     include_dirs  Include directories of the compiler
  @param[in,out] visited       Files already followed, in the directory they were found in
  @param[in,out] key           Disk cache key
  @param[in,out] files         Paths and identities of the included files
  @param[out]    is_followable Whether all the include directives were followed

  @return An error code: 0 - success, otherwise - failure
**/
static inline int JitCacheAddIncludes(Ceed ceed, const std::vector<JitInclude> &includes, const std::string &dir,
                                      const std::vector<std::string> &include_dirs, std::set<JitFileInDir> *visited, std::ostringstream *key,
                                      std::vector<std::pair<std::string, struct stat>> *files, bool *is_followable) {
  *is_followable = false;
  for (const JitInclude &include : includes) {
    std::vector<std::string> candidates;
    bool                     is_found = false;

    if (include.name[0] == '/') {
      candidates.push_back(include.name);
    } else {
      if (include.is_quoted) candidates.push_back(dir + include.name);
      for (const std::string &include_dir : include_dirs) {
        candidates.push_back(include_dir.empty() || include_dir.back() == '/' ? include_dir + include.name : include_dir + "/" + include.name);
      }
    }
    for (const std::string &path : candidates) {
      const std::string path_dir = path.substr(0, path.rfind('/') + 1);
      struct stat       file_stat, dir_stat;
      JitIncludedFile   file;
      bool              is_file_followable;

      if (stat(path.c_str(), &file_stat) != 0 || !S_ISREG(file_stat.st_mode)) continue;
      is_found = true;
      if (stat(path_dir.empty() ? "." : path_dir.c_str(), &dir_stat) != 0) return CEED_ERROR_SUCCESS;
      if (!visited->insert(JitFileInDir(file_stat.st_dev, file_stat.st_ino, dir_stat.st_dev, dir_stat.st_ino)).second) continue;
      CeedCallBackend(JitCacheGetFile(ceed, path, file_stat, &file));
      if (!file.is_followable) return CEED_ERROR_SUCCESS;
      *key << "file " << path << ' ' << file.hash << '\n';
      files->emplace_back(path, file.file_stat);
      CeedCallBackend(JitCacheAddIncludes(ceed, file.includes, path_dir, include_dirs, visited, key, files, &is_file_followable));
      if (!is_file_followable) return CEED_ERROR_SUCCESS;
    }
    // The compiler may not find a quoted header found nowhere either; an angle header found nowhere is the compiler's own
    if (include.is_quoted && !is_found) return CEED_ERROR_SUCCESS;
  }
  *is_followable = true;
  return CEED_ERROR_SUCCESS;
}

// Path, size, and modification time of the loaded library that defines `symbol`, or empty if unknown
static inline std::string JitCacheGetLibrary(const void *symbol) {
  Dl_info     info;
  struct stat file_stat;

  if (!dladdr(symbol, &info) || !info.dli_fname || stat(info.dli_fname, &file_stat) != 0) return "";
  return std::string(info.dli_fname) + " " + std::to_string(file_stat.st_size) + " " + std::to_string(file_stat.st_mtim.tv_sec) + "." +
         std::to_string(file_stat.st_mtim.tv_nsec);
}

// First line of an entry file
static inline std::string JitCacheEntryHeader(const std::string &key_hash, const char *binary, size_t size) {
  return std::string(jit_cache_format) + " " + key_hash + " " + JitCacheHash(binary, size) + "\n";
}

/**
  @brief Read the binary of an entry file

  @param[in]  ceed     `Ceed` context
  @param[in]  entry    JiT compilation, with the path of its entry file
  @param[out] binary   Binary of the entry
  @param[out] is_valid Whether the entry file exists and is valid

  @return An error code: 0 - success, otherwise - failure
**/
static inline int JitCacheReadEntry(Ceed ceed, const CeedJitCacheEntry &entry, std::string *binary, bool *is_valid) {
  std::string contents;
  struct stat file_stat;
  bool        is_read;

  *is_valid = false;
  CeedCallBackend(JitCacheReadFile(ceed, entry.path, true, &contents, &file_stat, &is_read));
  if (!is_read) return CEED_ERROR_SUCCESS;

  const size_t header_end = contents.find('\n');

  if (header_end == std::string::npos) return CEED_ERROR_SUCCESS;
  *binary   = contents.substr(header_end + 1);
  *is_valid = contents.compare(0, header_end + 1, JitCacheEntryHeader(entry.hash, binary->data(), binary->size())) == 0;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Get the state of a disk cache directory, checking once per process whether it is private

  @param[in]  ceed      `Ceed` context
  @param[in]  dir       Path of the directory
  @param[out] cache_dir State of the directory

  @return An error code: 0 - success, otherwise - failure
**/
static inline int JitCacheGetDir(Ceed ceed, const std::string &dir, JitCacheDir *cache_dir) {
  JitCacheState              &state = GetJitCacheState();
  std::lock_guard<std::mutex> lock(state.mutex);
  const auto                  cached = state.dirs.find(dir);

  if (cached != state.dirs.end()) {
    *cache_dir = cached->second;
  } else {
    const int   fd = open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    struct stat dir_stat;

    cache_dir->is_private  = fd >= 0 && fstat(fd, &dir_stat) == 0 && JitCacheIsPrivate(dir_stat);
    cache_dir->is_writable = cache_dir->is_private;
    if (fd >= 0) close(fd);
    state.dirs[dir] = *cache_dir;
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Look up a JiT compilation on disk, setting the path of its entry file if it can be cached on disk

  @param[in]     ceed            `Ceed` context
  @param[in]     compiler        Name and version of the compiler, or empty to not use the disk
  @param[in]     compiler_symbol Function of the compiler library, to identify it
  @param[in]     num_opts        Number of compiler options
  @param[in]     opts            Compiler options
  @param[in]     source          Source to compile
  @param[in]     name            Name of the kernel, for the entry file
  @param[in]     extension       File extension of the compiler, for the entry file
  @param[in,out] entry           JiT compilation

  @return An error code: 0 - success, otherwise - failure
**/
static inline int JitCacheFindOnDisk(Ceed ceed, const std::string &compiler, const void *compiler_symbol, int num_opts, const char *const *opts,
                                     const std::string &source, const char *name, const char *extension, CeedJitCacheEntry *entry) {
  // The compiler library does not change while the process runs
  static const std::string library = JitCacheGetLibrary(compiler_symbol);
  const char              *dir;
  JitCacheDir              cache_dir;
  std::ostringstream       key;
  std::vector<JitInclude>  includes;
  std::vector<std::string> include_dirs;
  std::set<JitFileInDir>   visited;
  std::string              binary;
  bool                     is_followable, is_valid;

  if (compiler.empty() || library.empty()) {
    CeedDebug(ceed, "Not using the JiT cache on disk: the compiler cannot be identified");
    return CEED_ERROR_SUCCESS;
  }
  CeedCallBackend(CeedTryGetCacheDir(ceed, &dir));
  if (dir) entry->dir = dir;
  CeedCallBackend(CeedRestoreCacheDir(ceed, &dir));
  if (entry->dir.empty()) {
    CeedDebug(ceed, "Not using the JiT cache on disk: the cache directory cannot be used");
    return CEED_ERROR_SUCCESS;
  }
  CeedCallBackend(JitCacheGetDir(ceed, entry->dir, &cache_dir));
  if (!cache_dir.is_private) {
    CeedDebug(ceed, "Not using the JiT cache on disk: the cache directory %s must belong to the user and not be writable by others",
              entry->dir.c_str());
    return CEED_ERROR_SUCCESS;
  }

  // Key: compiler, options and source (with their size, so that the end of the source is unambiguous), included files
  key << jit_cache_format << '\n' << "compiler " << compiler << '\n' << "library " << library << '\n';
  key << "input " << entry->key.size() << '\n' << entry->key << '\n';
  for (int i = 0; i < num_opts; i++) {
    if (!strncmp(opts[i], "-I", 2)) include_dirs.push_back(opts[i] + 2);
  }
  CeedCallBackend(JitCacheParseIncludes(ceed, source, &includes, &is_followable));
  if (is_followable) CeedCallBackend(JitCacheAddIncludes(ceed, includes, "", include_dirs, &visited, &key, &entry->files, &is_followable));
  if (!is_followable) {
    CeedDebug(ceed, "Not using the JiT cache on disk: an include directive cannot be followed");
    return CEED_ERROR_SUCCESS;
  }
  const std::string key_string = key.str();

  entry->hash = JitCacheHash(key_string.data(), key_string.size());
  entry->path = entry->dir + "/function_" + entry->hash + "_" + name + "." + extension;
  CeedCallBackend(JitCacheReadEntry(ceed, *entry, &binary, &is_valid));
  if (is_valid) {
    entry->binary           = std::make_shared<std::string>(std::move(binary));
    entry->is_found_on_disk = true;
  }
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Write data to a file

  @param[in]  ceed       `Ceed` context
  @param[in]  fd         File descriptor
  @param[in]  data       Data to write
  @param[in]  size       Size of the data
  @param[out] is_written Whether all the data was written

  @return An error code: 0 - success, otherwise - failure
**/
static inline int JitCacheWrite(Ceed ceed, int fd, const char *data, size_t size, bool *is_written) {
  *is_written = false;
  while (size > 0) {
    const ssize_t n = write(fd, data, size);

    if (n <= 0) return CEED_ERROR_SUCCESS;
    data += n;
    size -= static_cast<size_t>(n);
  }
  *is_written = true;
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Store a JiT compilation on disk, unless an included file changed or a valid entry already exists

  @param[in] ceed   `Ceed` context
  @param[in] entry  JiT compilation, with the path of its entry file
  @param[in] binary Compiled binary
  @param[in] size   Size of the compiled binary

  @return An error code: 0 - success, otherwise - failure
**/
static inline int JitCacheStoreOnDisk(Ceed ceed, const CeedJitCacheEntry &entry, const char *binary, size_t size) {
  JitCacheState    &state = GetJitCacheState();
  JitCacheDir       cache_dir;
  std::string       existing, tmp_path = entry.dir + "/.tmp_XXXXXX";
  const std::string header = JitCacheEntryHeader(entry.hash, binary, size);
  bool              is_valid, is_written;
  int               fd, error = 0;

  CeedCallBackend(JitCacheGetDir(ceed, entry.dir, &cache_dir));
  if (!cache_dir.is_writable) {
    CeedDebug(ceed, "Not storing the JiT binary on disk: %s is not writable", entry.dir.c_str());
    return CEED_ERROR_SUCCESS;
  }
  // The binary may not match the key if an included file changed, e.g. during the compilation
  for (const auto &file : entry.files) {
    struct stat file_stat;

    if (stat(file.first.c_str(), &file_stat) != 0 || !JitCacheIsSameFile(file.second, file_stat)) {
      CeedDebug(ceed, "Not storing the JiT binary on disk: %s changed", file.first.c_str());
      return CEED_ERROR_SUCCESS;
    }
  }
  // Another process may have stored it meanwhile; an entry found on disk which did not load is replaced
  if (!entry.is_found_on_disk) {
    CeedCallBackend(JitCacheReadEntry(ceed, entry, &existing, &is_valid));
    if (is_valid) {
      CeedDebug(ceed, "Not storing the JiT binary on disk: already stored");
      return CEED_ERROR_SUCCESS;
    }
  }

  // Write the entry to a temporary file (mkstemp creates it exclusively and private to the user), then publish it with
  // link(), which keeps a valid entry stored meanwhile by another process, or else replace the entry with rename()
  fd = mkstemp(&tmp_path[0]);
  if (fd < 0) {
    error = errno;
  } else {
    CeedCallBackend(JitCacheWrite(ceed, fd, header.data(), header.size(), &is_written));
    if (is_written) CeedCallBackend(JitCacheWrite(ceed, fd, binary, size, &is_written));
    if (!is_written) error = errno ? errno : EIO;
    if (close(fd) != 0 && !error) error = errno;
    if (!error) {
      if (!entry.is_found_on_disk && link(tmp_path.c_str(), entry.path.c_str()) == 0) {
        // Published
      } else if (!entry.is_found_on_disk && errno == EEXIST) {
        CeedCallBackend(JitCacheReadEntry(ceed, entry, &existing, &is_valid));
        if (is_valid) {
          error = EEXIST;  // stored meanwhile
        } else if (rename(tmp_path.c_str(), entry.path.c_str()) != 0) {
          error = errno;
        }
      } else if (rename(tmp_path.c_str(), entry.path.c_str()) != 0) {
        error = errno;
      }
    }
    unlink(tmp_path.c_str());
  }
  if (!error) {
    CeedDebug(ceed, "Stored JiT binary on disk: %s", entry.path.c_str());
  } else {
    // Stop writing to a directory which is read-only or full
    if (error == EROFS || error == EACCES || error == EPERM || error == ENOSPC || error == EDQUOT) {
      std::lock_guard<std::mutex> lock(state.mutex);

      state.dirs[entry.dir].is_writable = false;
    }
    CeedDebug(ceed, "Not storing the JiT binary on disk: %s", strerror(error));
  }
  return CEED_ERROR_SUCCESS;
}

}  // namespace internal
}  // namespace ceed

/**
  @brief Look up a JiT compilation in memory, then on disk

  If `binary` is set to `NULL`, or if the cached binary does not load, the compiled binary must be stored with @ref CeedJitCacheStore().

  @param[in]  ceed            `Ceed` context
  @param[in]  compiler        Name and version of the compiler, or empty to not use the disk
  @param[in]  compiler_symbol Function of the compiler library, to identify it
  @param[in]  num_opts        Number of compiler options
  @param[in]  opts            Compiler options
  @param[in]  source          Source to compile
  @param[in]  name            Name of the kernel, for the entry file
  @param[in]  extension       File extension of the compiler, e.g. "nvrtc", for the entry file
  @param[out] entry           Lookup state, to pass to @ref CeedJitCacheStore()
  @param[out] binary          Cached binary, or `NULL` if not found

  @return An error code: 0 - success, otherwise - failure
**/
static inline int CeedJitCacheFind(Ceed ceed, const std::string &compiler, const void *compiler_symbol, int num_opts, const char *const *opts,
                                   const std::string &source, const char *name, const char *extension, CeedJitCacheEntry *entry,
                                   const std::string **binary) {
  ceed::internal::JitCacheState &state = ceed::internal::GetJitCacheState();

  *binary = nullptr;
  for (int i = 0; i < num_opts; i++) entry->key.append(opts[i]).push_back('\0');
  entry->key += source;
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    const auto                  cached = state.binaries.find(entry->key);

    if (cached != state.binaries.end()) entry->binary = cached->second;
  }
  if (entry->binary) {
    CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "---------- USING CACHED JiT BINARY: %s ----------\n", name);
  } else {
    CeedCallBackend(ceed::internal::JitCacheFindOnDisk(ceed, compiler, compiler_symbol, num_opts, opts, source, name, extension, entry));
    // Failures to use the disk, e.g. a missing entry, are not errors
    errno = 0;
    if (entry->binary) {
      std::lock_guard<std::mutex> lock(state.mutex);

      state.binaries[entry->key] = entry->binary;
      CeedDebug256(ceed, CEED_DEBUG_COLOR_SUCCESS, "---------- USING JiT BINARY CACHED ON DISK: %s ----------\n", name);
      CeedDebug(ceed, "Entry file: %s", entry->path.c_str());
    }
  }
  if (entry->binary) *binary = entry->binary.get();
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Store a successful JiT compilation not found by @ref CeedJitCacheFind(), or found but failing to load

  @param[in]     ceed   `Ceed` context
  @param[in,out] entry  Lookup state from @ref CeedJitCacheFind()
  @param[in]     binary Compiled binary
  @param[in]     size   Size of the compiled binary

  @return An error code: 0 - success, otherwise - failure
**/
static inline int CeedJitCacheStore(Ceed ceed, CeedJitCacheEntry *entry, const char *binary, size_t size) {
  ceed::internal::JitCacheState &state = ceed::internal::GetJitCacheState();

  if (!entry->path.empty()) {
    CeedCallBackend(ceed::internal::JitCacheStoreOnDisk(ceed, *entry, binary, size));
    // Failures to use the disk are not errors
    errno = 0;
  }
  {
    std::lock_guard<std::mutex> lock(state.mutex);

    state.binaries[entry->key] = std::make_shared<std::string>(binary, size);
  }
  return CEED_ERROR_SUCCESS;
}
