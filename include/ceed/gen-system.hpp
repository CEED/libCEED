// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

/// @file
/// Public header for system call utilities of libCEED
#pragma once

#include <ceed/backend.h>

#include <cerrno>
#include <cstring>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <unistd.h>
#include <sys/wait.h>

// In internal namespace to avoid polluting files that include this
namespace ceed {
namespace internal {
struct CeedProcessResult {
  std::string stdout_output;
  std::string stderr_output;
  int         exit_status{0};
};

// RAII helper to ensure file descriptors are closed safely
struct ScopedFileDescriptor {
  int fd                 = -1;
  ScopedFileDescriptor() = default;
  explicit ScopedFileDescriptor(int f) : fd(f) {}
  ~ScopedFileDescriptor() { reset(); }

  void reset(int new_fd = -1) {
    if (fd >= 0) {
      close(fd);
    }
    fd = new_fd;
  }

  int get() const { return fd; }
  int release() {
    int temp = fd;
    fd       = -1;
    return temp;
  }
};

void set_nonblocking(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  if (flags != -1) fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

void drain_pipe(ScopedFileDescriptor &pipe_fd, std::string &destination) {
  char buffer[8192];

  while (true) {
    ssize_t bytes = read(pipe_fd.get(), buffer, sizeof(buffer));

    if (bytes > 0) {
      destination.append(buffer, static_cast<std::size_t>(bytes));
    } else if (bytes == 0) {
      pipe_fd.reset();
      break;
    } else {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        break;
      }
      // LCOV_EXCL_START
      if (errno == EINTR) {
        continue;
      }
      pipe_fd.reset();
      break;
      // LCOV_EXCL_STOP
    }
  }
}

int CeedCallSystem_Internal(Ceed ceed, const std::string &command, const std::string &input_data, CeedProcessResult &result) {
  int in_pipe[2]  = {-1, -1};
  int out_pipe[2] = {-1, -1};
  int err_pipe[2] = {-1, -1};
  // Use pipe2 with O_CLOEXEC to prevent leaking FDs if other threads fork/exec
#if defined(__linux__)
  if (pipe2(in_pipe, O_CLOEXEC) != 0 || pipe2(out_pipe, O_CLOEXEC) != 0 || pipe2(err_pipe, O_CLOEXEC) != 0) {
    // LCOV_EXCL_START
    result.exit_status   = 127;
    result.stderr_output = "Failed to create pipes\n";
    return CEED_ERROR_SUCCESS;
    // LCOV_EXCL_STOP
  }
#else
  // MacOS doesn't have pipe2
  if (pipe(in_pipe) != 0 || pipe(out_pipe) != 0 || pipe(err_pipe) != 0) {
    // LCOV_EXCL_START
    result.exit_status   = 127;
    result.stderr_output = "Failed to create pipes\n";
    return CEED_ERROR_SUCCESS;
    // LCOV_EXCL_STOP
  }
  fcntl(in_pipe[0], F_SETFD, FD_CLOEXEC);
  fcntl(in_pipe[1], F_SETFD, FD_CLOEXEC);
  fcntl(out_pipe[0], F_SETFD, FD_CLOEXEC);
  fcntl(out_pipe[1], F_SETFD, FD_CLOEXEC);
  fcntl(err_pipe[0], F_SETFD, FD_CLOEXEC);
  fcntl(err_pipe[1], F_SETFD, FD_CLOEXEC);
#endif

  ScopedFileDescriptor parent_in_read(in_pipe[0]);
  ScopedFileDescriptor parent_in_write(in_pipe[1]);
  ScopedFileDescriptor parent_out_read(out_pipe[0]);
  ScopedFileDescriptor parent_out_write(out_pipe[1]);
  ScopedFileDescriptor parent_err_read(err_pipe[0]);
  ScopedFileDescriptor parent_err_write(err_pipe[1]);

  pid_t pid = fork();

  if (pid < 0) {
    // LCOV_EXCL_START
    result.exit_status   = 127;
    result.stderr_output = "Failed to fork process\n";
    return CEED_ERROR_SUCCESS;
    // LCOV_EXCL_STOP
  }

  if (pid == 0) {
    // In the child process (essentially the same as system() behavior)
    // Note: uses _exit to avoid side-effects of exit()
    // Restore default SIGPIPE behavior in child
    signal(SIGPIPE, SIG_DFL);

    // Redirect stdin to read-end of input pipe
    if (dup2(parent_in_read.get(), STDIN_FILENO) == -1) _exit(127);
    // Redirect stdout to write-end of output pipe
    if (dup2(parent_out_write.get(), STDOUT_FILENO) == -1) _exit(127);
    // Redirect stderr to write-end of error pipe
    if (dup2(parent_err_write.get(), STDERR_FILENO) == -1) _exit(127);

    // Close all pipe descriptors (dup2 duplicates them to 0 and 1)
    parent_in_read.reset();
    parent_in_write.reset();
    parent_out_read.reset();
    parent_out_write.reset();
    parent_err_read.reset();
    parent_err_write.reset();

    execl("/bin/sh", "sh", "-c", command.c_str(), nullptr);
    // LCOV_EXCL_START
    _exit(127);
    // LCOV_EXCL_STOP
  }

  // In the parent process
  // Close endpoints owned exclusively by the child
  parent_in_read.reset();
  parent_out_write.reset();
  parent_err_write.reset();

  // Set parent endpoints to non-blocking mode so that poll works
  set_nonblocking(parent_in_write.get());
  set_nonblocking(parent_out_read.get());
  set_nonblocking(parent_err_read.get());

  // Ignore SIGPIPE so writing to a terminated child returns EPIPE instead of killing the process
  struct sigaction sa{}, old_sa{};

  sa.sa_handler = SIG_IGN;
  sigaction(SIGPIPE, &sa, &old_sa);

  std::string stdout_output;
  std::string stderr_output;

  size_t input_bytes_written = 0;
  bool   write_closed        = input_data.empty();

  if (write_closed) {
    parent_in_write.reset();
  }

  bool success = true;

  // Poll over file descriptors until both are closed
  while (parent_out_read.get() >= 0 || parent_err_read.get() >= 0 || parent_in_write.get() >= 0) {
    struct pollfd polling_file_descriptors[3];
    nfds_t        num_active_descriptors = 0;
    int           stdin_idx              = -1;
    int           stdout_idx             = -1;
    int           stderr_idx             = -1;

    if (parent_in_write.get() >= 0) {
      stdin_idx                                   = num_active_descriptors++;
      polling_file_descriptors[stdin_idx].fd      = parent_in_write.get();
      polling_file_descriptors[stdin_idx].events  = POLLOUT;
      polling_file_descriptors[stdin_idx].revents = 0;
    }
    if (parent_out_read.get() >= 0) {
      stdout_idx                                   = num_active_descriptors++;
      polling_file_descriptors[stdout_idx].fd      = parent_out_read.get();
      polling_file_descriptors[stdout_idx].events  = POLLIN;
      polling_file_descriptors[stdout_idx].revents = 0;
    }
    if (parent_err_read.get() >= 0) {
      stderr_idx                                   = num_active_descriptors++;
      polling_file_descriptors[stderr_idx].fd      = parent_err_read.get();
      polling_file_descriptors[stderr_idx].events  = POLLIN;
      polling_file_descriptors[stderr_idx].revents = 0;
    }

    int poll_err = poll(polling_file_descriptors, num_active_descriptors, -1);

    if (poll_err < 0) {
      // LCOV_EXCL_START
      if (errno == EINTR) continue;
      success = false;
      break;
      // LCOV_EXCL_STOP
    }

    // Handle writing to child's stdin
    // Note: this potentially happens in blocks, thus we track the amount written and offset the buffer
    if (stdin_idx != -1 && (polling_file_descriptors[stdin_idx].revents & (POLLOUT | POLLERR | POLLHUP))) {
      if (polling_file_descriptors[stdin_idx].revents & POLLOUT) {
        const char *buf     = input_data.data() + input_bytes_written;
        size_t      count   = input_data.size() - input_bytes_written;
        ssize_t     written = write(parent_in_write.get(), buf, count);

        if (written > 0) {
          input_bytes_written += written;
          if (input_bytes_written >= input_data.size()) {
            // Finished writing input: close pipe to send EOF to child
            parent_in_write.reset();
          }
        } else if (written < 0 && (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)) {
          // LCOV_EXCL_START
          // EPIPE or other error: child probably exited early
          parent_in_write.reset();
          // LCOV_EXCL_STOP
        }
      } else {
        // LCOV_EXCL_START
        // Pipe error or hangup
        parent_in_write.reset();
        // LCOV_EXCL_STOP
      }
    }

    // Handle reading from child's stdout
    if (stdout_idx != -1 && (polling_file_descriptors[stdout_idx].revents & (POLLIN | POLLHUP | POLLERR))) {
      if (polling_file_descriptors[stdout_idx].revents & POLLIN) {
        drain_pipe(parent_out_read, stdout_output);
      }
      if (polling_file_descriptors[stdout_idx].revents & POLLHUP) {
        parent_out_read.reset();
      }
    }

    // Handle reading from child's stderr
    if (stderr_idx != -1 && (polling_file_descriptors[stderr_idx].revents & (POLLIN | POLLHUP | POLLERR))) {
      if (polling_file_descriptors[stderr_idx].revents & POLLIN) {
        drain_pipe(parent_err_read, stderr_output);
      }
      if (polling_file_descriptors[stderr_idx].revents & POLLHUP) {
        parent_err_read.reset();
      }
    }
  }

  // Failed command execution, error
  if (!success) {
    // LCOV_EXCL_START
    sigaction(SIGPIPE, &old_sa, nullptr);
    kill(pid, SIGTERM);
    waitpid(pid, nullptr, 0);
    result.exit_status   = 127;
    result.stderr_output = std::string("System command '") + command + "' failed: " + strerror(errno);
    return CEED_ERROR_SUCCESS;
    // LCOV_EXCL_STOP
  }

  // Reset SIGPIPE signal handler
  sigaction(SIGPIPE, &old_sa, nullptr);

  // Wait for the child process to terminate
  int status = 0;

  while (waitpid(pid, &status, 0) < 0) {
    // LCOV_EXCL_START
    if (errno != EINTR) {
      result.exit_status   = -1;
      result.stderr_output = std::string("Waiting for child process failed: ") + strerror(errno);
      return CEED_ERROR_SUCCESS;
    }
    // LCOV_EXCL_STOP
  }
  result.stdout_output = std::move(stdout_output);
  result.stderr_output = std::move(stderr_output);
  if (WIFEXITED(status)) {
    result.exit_status = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    // LCOV_EXCL_START
    result.exit_status = 128 + WTERMSIG(status);
    // LCOV_EXCL_STOP
  }
  return CEED_ERROR_SUCCESS;
}
}  // namespace internal
}  // namespace ceed

/// Struct storing the stderr and stdout from a system call
struct CeedCallSystemResult {
  bool        is_success = false;
  std::string output{};
  std::string error{};
};

/**
  @brief Call to shell with `stdin` from a string, allowed to fail

  @param[in]  ceed    `Ceed` context
  @param[in]  command Command to run
  @param[in]  message Message describing the action of the command
  @param[in]  input   String to pipe to `stdin`
  @param[out] result  Reference to `CeedCallSystemResult` to store the output, error, and success status

  @return An error code: 0 - success, otherwise - failure
**/
int CeedCallSystemUnchecked(Ceed ceed, const std::string &command, const std::string &message, const std::string &input,
                            CeedCallSystemResult &result) {
  ceed::internal::CeedProcessResult process_result;

  CeedDebug(ceed, "Running command:\n$ %s", command.c_str());
  CeedCall(CeedCallSystem_Internal(ceed, command, input, process_result));

  result.is_success = process_result.exit_status == 0;
  result.error      = std::move(process_result.stderr_output);
  result.output     = std::move(process_result.stdout_output);
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Call to shell, allowed to fail

  @param[in]  ceed    `Ceed` context
  @param[in]  command Command to run
  @param[in]  message Message describing the action of the command
  @param[out] result  Reference to `CeedCallSystemResult` to store the output, error, and success status

  @return An error code: 0 - success, otherwise - failure
**/
int CeedCallSystemUnchecked(Ceed ceed, const std::string &command, const std::string &message, CeedCallSystemResult &result) {
  CeedCall(CeedCallSystemUnchecked(ceed, command, message, std::string{}, result));
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Call to shell with `stdin` from a string, errors if command fails

  @param[in]  ceed    `Ceed` context
  @param[in]  command Command to run
  @param[in]  message Message describing the action of the command
  @param[in]  input   String to pipe to `stdin`
  @param[out] result  Reference to `CeedCallSystemResult` to store the output, error, and success status

  @return An error code: 0 - success, otherwise - failure
**/
int CeedCallSystem(Ceed ceed, const std::string &command, const std::string &message, const std::string &input, CeedCallSystemResult &result) {
  CeedCall(CeedCallSystemUnchecked(ceed, command, message, input, result));
  CeedCheck(result.is_success, ceed, CEED_ERROR_MAJOR, "Failed to %s\ncommand:\n$ %s\nerror:\n%s", message.c_str(), command.c_str(),
            result.error.c_str());
  return CEED_ERROR_SUCCESS;
}

/**
  @brief Call to shell, errors if command fails

  @param[in]  ceed    `Ceed` context
  @param[in]  command Command to run
  @param[in]  message Message describing the action of the command
  @param[out] result  Reference to `CeedCallSystemResult` to store the output, error, and success status

  @return An error code: 0 - success, otherwise - failure
**/
int CeedCallSystem(Ceed ceed, const std::string &command, const std::string &message, CeedCallSystemResult &result) {
  CeedCall(CeedCallSystem(ceed, command, message, std::string{}, result));
  return CEED_ERROR_SUCCESS;
}
