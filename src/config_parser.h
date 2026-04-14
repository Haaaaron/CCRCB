#pragma once

#include "benchmark_lattice.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

struct RunConfig {
  std::vector<size_t> volume;
  int iters;
  int spill_words;
  int warmup;
  int repeats;
  std::vector<int> comm_mask;
  CommBackend backend;
  std::string backend_name;
};

/**
 * @brief Converts string to CommBackend enum. Throws if invalid.
 */
inline CommBackend parse_backend(const std::string &str) {
  if (str == "NCCL")
    return CommBackend::NCCL;
  if (str == "MPI")
    return CommBackend::CUDA_AWARE_MPI;
  if (str == "NVSHMEM")
    return CommBackend::NVSHMEM;
  throw std::invalid_argument("Parser Error: Unknown backend string '" + str +
                              "'");
}

/**
 * @brief Parses (x,y,z) tuples. Throws if formatting is invalid.
 */
template <typename T>
inline std::vector<T> parse_tuple(const std::string &str) {
  std::vector<T> result;
  if (str.length() < 2 || str.front() != '(' || str.back() != ')') {
    throw std::runtime_error("Parser Error: Malformed tuple '" + str +
                             "'. Expected (val,val,...)");
  }

  std::string inner = str.substr(1, str.length() - 2);
  std::stringstream ss(inner);
  std::string item;

  while (std::getline(ss, item, ',')) {
    if (item.empty())
      continue;
    try {
      if constexpr (std::is_same_v<T, size_t>) {
        result.push_back(std::stoull(item));
      } else {
        result.push_back(std::stoi(item));
      }
    } catch (...) {
      throw std::runtime_error(
          "Parser Error: Could not convert tuple element '" + item +
          "' to numeric type.");
    }
  }
  return result;
}

/**
 * @brief Parses the configuration file. Uses stream exceptions for strict
 * validation.
 */
inline std::vector<RunConfig> parse_run_file(const std::string &filename,
                                             int rank) {
  std::vector<RunConfig> configs;
  std::ifstream infile(filename);

  if (!infile.is_open()) {
    if (rank == 0)
      std::cerr << "CRITICAL: Cannot open config file: " << filename << "\n";
    return configs;
  }

  std::string line;
  int line_num = 0;

  while (std::getline(infile, line)) {
    line_num++;

    // Skip empty lines or comments
    if (line.empty() || line.find_first_not_of(" \t\n\r") == std::string::npos)
      continue;
    size_t first = line.find_first_not_of(" \t");
    if (line[first] == '#')
      continue;

    std::stringstream ss(line);

    // ENABLE EXCEPTIONS for this line: Throws on type mismatch or unexpected
    // end of line
    ss.exceptions(std::ios::failbit | std::ios::badbit);

    std::string vol_str, comm_str, backend_str;
    int iters, spill, warmup, repeats;

    try {
      // Extraction must follow the exact order in runs.txt
      ss >> vol_str >> iters >> spill >> warmup >> repeats >> comm_str >>
          backend_str;

      RunConfig cfg;
      cfg.volume = parse_tuple<size_t>(vol_str);
      cfg.iters = iters;
      cfg.spill_words = spill;
      cfg.warmup = warmup;
      cfg.repeats = repeats;
      cfg.comm_mask = parse_tuple<int>(comm_str);
      cfg.backend = parse_backend(backend_str);
      cfg.backend_name = backend_str;

      configs.push_back(cfg);

    } catch (const std::ios_base::failure &e) {
      if (rank == 0) {
        std::cerr << "CRITICAL: Data mismatch on line " << line_num << " of "
                  << filename << "\n";
        std::cerr << "Content: " << line << "\n";
        std::cerr << "Ensure numeric columns (iters, spill, etc.) do not "
                     "contain text.\n";
      }
      throw; // Re-throw to stop execution
    } catch (const std::exception &e) {
      if (rank == 0) {
        std::cerr << "CRITICAL: Logic error on line " << line_num << ": "
                  << e.what() << "\n";
      }
      throw;
    }
  }

  if (rank == 0) {
    printf("Successfully loaded %zu configurations from %s\n", configs.size(),
           filename.c_str());
  }

  return configs;
}
