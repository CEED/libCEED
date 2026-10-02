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
#include <spawn.h>
#include <unistd.h>
#include <sys/wait.h>

extern char **environ;

// In internal namespace to avoid polluting files that include this
namespace ceed {
namespace internal {
struct CeedProcessResult {
  std::string stdout_output;
  std::string stderr_output;
  int         exit_status{0};
};

static inline void set_nonblocking(int fd) {
  int flags = fcntl(fd, F_GETFL, 0);
  if (flags != -1) fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

static inline void reset(int &fd) {
  if (fd >= 0) {
    close(fd);
  }
  fd = -1;
}

static inline void drain_pipe(int &pipe_fd, std::string &destination) {
  char buffer[8192];

  while (true) {
    ssize_t bytes = read(pipe_fd, buffer, sizeof(buffer));

    if (bytes > 0) {
      destination.append(buffer, static_cast<std::size_t>(bytes));
    } else if (bytes == 0) {
      reset(pipe_fd);
      break;
    } else {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        break;
      }
      // LCOV_EXCL_START
      if (errno == EINTR) {
        continue;
      }
      reset(pipe_fd);
      break;
      // LCOV_EXCL_STOP
    }
  }
}

static inline int CeedCallSystem_Internal(Ceed ceed, const std::string &command, const std::string &input_data, CeedProcessResult &result) {
  int               in_pipe[2]  = {-1, -1};
  int               out_pipe[2] = {-1, -1};
  int               err_pipe[2] = {-1, -1};
  const char *const argv[]      = {
      "/bin/sh",
      "-c",
      command.c_str(),
      NULL,
  };
  bool write_closed = input_data.empty();

  // Use pipe2 with O_CLOEXEC to prevent leaking FDs if other threads fork/exec
#if defined(__linux__)
  if (pipe2(out_pipe, O_CLOEXEC) != 0 || pipe2(err_pipe, O_CLOEXEC) != 0) {
    // LCOV_EXCL_START
    result.exit_status   = 127;
    result.stderr_output = "Failed to create pipes\n";
    return CEED_ERROR_SUCCESS;
    // LCOV_EXCL_STOP
  }
  if (!write_closed) {
    if (pipe2(in_pipe, O_CLOEXEC) != 0) {
      // LCOV_EXCL_START
      result.exit_status   = 127;
      result.stderr_output = "Failed to create pipes\n";
      return CEED_ERROR_SUCCESS;
      // LCOV_EXCL_STOP
    }
  }
#else
  // MacOS doesn't have pipe2
  if (pipe(out_pipe) != 0 || pipe(err_pipe) != 0) {
    // LCOV_EXCL_START
    result.exit_status   = 127;
    result.stderr_output = "Failed to create pipes\n";
    return CEED_ERROR_SUCCESS;
    // LCOV_EXCL_STOP
  }

  if (!write_closed) {
    if (pipe(in_pipe) != 0) {
      // LCOV_EXCL_START
      result.exit_status   = 127;
      result.stderr_output = "Failed to create pipes\n";
      return CEED_ERROR_SUCCESS;
      // LCOV_EXCL_STOP
    }
    fcntl(in_pipe[0], F_SETFD, FD_CLOEXEC);
    fcntl(in_pipe[1], F_SETFD, FD_CLOEXEC);
  }
  fcntl(out_pipe[0], F_SETFD, FD_CLOEXEC);
  fcntl(out_pipe[1], F_SETFD, FD_CLOEXEC);
  fcntl(err_pipe[0], F_SETFD, FD_CLOEXEC);
  fcntl(err_pipe[1], F_SETFD, FD_CLOEXEC);
#endif

  posix_spawn_file_actions_t actions;
  pid_t                      pid;

  posix_spawn_file_actions_init(&actions);
  posix_spawn_file_actions_adddup2(&actions, out_pipe[1], STDOUT_FILENO);
  posix_spawn_file_actions_adddup2(&actions, err_pipe[1], STDERR_FILENO);
  posix_spawn_file_actions_addclose(&actions, out_pipe[0]);
  posix_spawn_file_actions_addclose(&actions, out_pipe[1]);
  posix_spawn_file_actions_addclose(&actions, err_pipe[0]);
  posix_spawn_file_actions_addclose(&actions, err_pipe[1]);
  if (!write_closed) {
    posix_spawn_file_actions_adddup2(&actions, in_pipe[0], STDIN_FILENO);
    posix_spawn_file_actions_addclose(&actions, in_pipe[0]);
    posix_spawn_file_actions_addclose(&actions, in_pipe[1]);
  }

  if (posix_spawn(&pid, "/bin/sh", &actions, NULL, (char *const *)argv, environ) < 0) {
    // LCOV_EXCL_START
    result.exit_status   = 127;
    result.stderr_output = "Failed to fork process\n";
    posix_spawn_file_actions_destroy(&actions);
    reset(in_pipe[0]), reset(in_pipe[1]);
    reset(out_pipe[0]), reset(out_pipe[1]);
    reset(err_pipe[0]), reset(err_pipe[1]);
    return CEED_ERROR_SUCCESS;
    // LCOV_EXCL_STOP
  }
  posix_spawn_file_actions_destroy(&actions);

  struct sigaction sa{}, old_sa{};

  sa.sa_handler = SIG_IGN;
  sigaction(SIGPIPE, &sa, &old_sa);

  bool        success = true;
  std::string stdout_output;
  std::string stderr_output;

  // In the parent process
  // Close endpoints owned exclusively by the child
  reset(in_pipe[0]);
  reset(out_pipe[1]);
  reset(err_pipe[1]);

  // Set parent endpoints to non-blocking mode so that poll works
  if (!write_closed) set_nonblocking(in_pipe[1]);
  set_nonblocking(out_pipe[0]);
  set_nonblocking(err_pipe[0]);

  // Ignore SIGPIPE so writing to a terminated child returns EPIPE instead of killing the process
  size_t input_bytes_written = 0;

  // Poll over file descriptors until both are closed
  while (out_pipe[0] >= 0 || err_pipe[0] >= 0 || in_pipe[1] >= 0) {
    struct pollfd polling_fds[3];
    nfds_t        num_active_descriptors = 0;
    int           stdin_idx = -1, stdout_idx = -1, stderr_idx = -1;

    if (in_pipe[1] >= 0) {
      stdin_idx                      = num_active_descriptors++;
      polling_fds[stdin_idx].fd      = in_pipe[1];
      polling_fds[stdin_idx].events  = POLLOUT;
      polling_fds[stdin_idx].revents = 0;
    }
    if (out_pipe[0] >= 0) {
      stdout_idx                      = num_active_descriptors++;
      polling_fds[stdout_idx].fd      = out_pipe[0];
      polling_fds[stdout_idx].events  = POLLIN;
      polling_fds[stdout_idx].revents = 0;
    }
    if (err_pipe[0] >= 0) {
      stderr_idx                      = num_active_descriptors++;
      polling_fds[stderr_idx].fd      = err_pipe[0];
      polling_fds[stderr_idx].events  = POLLIN;
      polling_fds[stderr_idx].revents = 0;
    }

    int poll_err = poll(polling_fds, num_active_descriptors, -1);

    if (poll_err < 0) {
      // LCOV_EXCL_START
      if (errno == EINTR) continue;
      success = false;
      break;
      // LCOV_EXCL_STOP
    }

    // Handle writing to child's stdin
    // Note: this potentially happens in blocks, thus we track the amount written and offset the buffer
    if (stdin_idx != -1 && (polling_fds[stdin_idx].revents & (POLLOUT | POLLERR | POLLHUP))) {
      if (polling_fds[stdin_idx].revents & POLLOUT) {
        const char *buf     = input_data.data() + input_bytes_written;
        size_t      count   = input_data.size() - input_bytes_written;
        ssize_t     written = write(in_pipe[1], buf, count);

        if (written > 0) {
          input_bytes_written += written;
          // Finished writing input: close pipe to send EOF to child
          if (input_bytes_written >= input_data.size()) reset(in_pipe[1]);
        } else if (written < 0 && (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)) {
          // LCOV_EXCL_START
          // EPIPE or other error: child probably exited early
          reset(in_pipe[1]);
          // LCOV_EXCL_STOP
        }
      } else {
        // LCOV_EXCL_START
        // Pipe error or hangup
        reset(in_pipe[1]);
        // LCOV_EXCL_STOP
      }
    }

    // Handle reading from child's stdout
    if (stdout_idx != -1 && (polling_fds[stdout_idx].revents & (POLLIN | POLLHUP | POLLERR))) {
      if (polling_fds[stdout_idx].revents & POLLIN) drain_pipe(out_pipe[0], stdout_output);
      if (polling_fds[stdout_idx].revents & POLLHUP) reset(out_pipe[0]);
    }

    // Handle reading from child's stderr
    if (stderr_idx != -1 && (polling_fds[stderr_idx].revents & (POLLIN | POLLHUP | POLLERR))) {
      if (polling_fds[stderr_idx].revents & POLLIN) drain_pipe(err_pipe[0], stderr_output);
      if (polling_fds[stderr_idx].revents & POLLHUP) reset(err_pipe[0]);
    }
  }

  // Reset SIGPIPE signal handler
  sigaction(SIGPIPE, &old_sa, nullptr);

  // Failed command execution, error
  if (!success) {
    // LCOV_EXCL_START
    kill(pid, SIGTERM);
    waitpid(pid, nullptr, 0);
    result.exit_status   = 127;
    result.stderr_output = std::string("System command '") + command + "' failed: " + strerror(errno);
    reset(in_pipe[0]), reset(in_pipe[1]);
    reset(out_pipe[0]), reset(out_pipe[1]);
    reset(err_pipe[0]), reset(err_pipe[1]);
    return CEED_ERROR_SUCCESS;
    // LCOV_EXCL_STOP
  }

  // Wait for the child process to terminate
  int status = 0;

  while (waitpid(pid, &status, 0) < 0) {
    // LCOV_EXCL_START
    if (errno != EINTR) {
      result.exit_status   = 127;
      result.stderr_output = std::string("Waiting for child process failed: ") + strerror(errno);
      reset(in_pipe[0]), reset(in_pipe[1]);
      reset(out_pipe[0]), reset(out_pipe[1]);
      reset(err_pipe[0]), reset(err_pipe[1]);
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
  reset(in_pipe[0]), reset(in_pipe[1]);
  reset(out_pipe[0]), reset(out_pipe[1]);
  reset(err_pipe[0]), reset(err_pipe[1]);
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
static inline int CeedCallSystemUnchecked(Ceed ceed, const std::string &command, const std::string &message, const std::string &input,
                                          CeedCallSystemResult &result) {
  ceed::internal::CeedProcessResult process_result;

  CeedDebug(ceed, "Running command:\n$ %s", command.c_str());
  CeedCall(CeedCallSystem_Internal(ceed, command, input, process_result));

  result.is_success = process_result.exit_status == 0;
  result.error      = std::move(process_result.stderr_output);
  result.output     = std::move(process_result.stdout_output);
  if (!result.is_success) {
    CeedDebug(ceed, "Failed running command.\nStatus code: %d\nError: %s", process_result.exit_status, result.error.c_str());
  }
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
static inline int CeedCallSystemUnchecked(Ceed ceed, const std::string &command, const std::string &message, CeedCallSystemResult &result) {
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
static inline int CeedCallSystem(Ceed ceed, const std::string &command, const std::string &message, const std::string &input,
                                 CeedCallSystemResult &result) {
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
static inline int CeedCallSystem(Ceed ceed, const std::string &command, const std::string &message, CeedCallSystemResult &result) {
  CeedCall(CeedCallSystem(ceed, command, message, std::string{}, result));
  return CEED_ERROR_SUCCESS;
}
