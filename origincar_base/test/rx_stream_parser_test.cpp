#include "origincar_base/rx_stream_parser.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using origincar_base_rx::Frame;
using origincar_base_rx::RxStreamParser;

namespace
{

Frame make_frame(uint8_t seed)
{
  Frame frame{};
  frame[0] = origincar_base_rx::kFrameHeader;
  for (std::size_t index = 1; index < 22; ++index) {
    frame[index] = static_cast<uint8_t>(seed + index * 3);
    if (frame[index] == origincar_base_rx::kFrameHeader) {
      frame[index] ^= 0x20;
    }
  }
  uint8_t checksum = 0;
  for (std::size_t index = 0; index < 22; ++index) {
    checksum ^= frame[index];
  }
  frame[22] = checksum;
  frame[23] = origincar_base_rx::kFrameTail;
  return frame;
}

void expect(bool condition, const std::string & message)
{
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void append_frame(RxStreamParser & parser, const Frame & frame)
{
  parser.append(frame.data(), frame.size());
}

void expect_frame(RxStreamParser & parser, const Frame & expected)
{
  Frame actual{};
  expect(parser.pop_frame(actual), "expected a parsed frame");
  expect(actual == expected, "parsed frame differs from input");
}

template<typename Function>
void run(const char * name, Function function, int & passed)
{
  function();
  ++passed;
  std::cout << "PASS " << name << '\n';
}

}  // namespace

int main()
{
  int passed = 0;

  run("single valid frame", [&]() {
    RxStreamParser parser;
    const auto frame = make_frame(1);
    append_frame(parser, frame);
    expect_frame(parser, frame);
  }, passed);

  run("two consecutive frames", [&]() {
    RxStreamParser parser;
    const auto first = make_frame(2);
    const auto second = make_frame(3);
    std::vector<uint8_t> bytes(first.begin(), first.end());
    bytes.insert(bytes.end(), second.begin(), second.end());
    parser.append(bytes.data(), bytes.size());
    expect_frame(parser, first);
    expect_frame(parser, second);
  }, passed);

  run("frame split in two", [&]() {
    RxStreamParser parser;
    const auto frame = make_frame(4);
    parser.append(frame.data(), 7);
    expect(!parser.has_frame(), "partial frame parsed too early");
    parser.append(frame.data() + 7, frame.size() - 7);
    expect_frame(parser, frame);
  }, passed);

  run("frame split many times", [&]() {
    RxStreamParser parser;
    const auto frame = make_frame(5);
    for (const auto byte : frame) {
      parser.append(&byte, 1);
    }
    expect_frame(parser, frame);
  }, passed);

  run("one garbage byte", [&]() {
    RxStreamParser parser;
    const uint8_t garbage = 0x11;
    parser.append(&garbage, 1);
    const auto frame = make_frame(6);
    append_frame(parser, frame);
    expect_frame(parser, frame);
    expect(parser.stats().discarded_bytes == 1, "garbage count mismatch");
  }, passed);

  run("twenty-three garbage bytes", [&]() {
    RxStreamParser parser;
    std::vector<uint8_t> garbage(23, 0x22);
    parser.append(garbage.data(), garbage.size());
    const auto frame = make_frame(7);
    append_frame(parser, frame);
    expect_frame(parser, frame);
    expect(parser.stats().discarded_bytes == 23, "garbage count mismatch");
  }, passed);

  run("start in previous frame middle", [&]() {
    RxStreamParser parser;
    const auto partial = make_frame(8);
    const auto complete = make_frame(9);
    parser.append(partial.data() + 11, partial.size() - 11);
    append_frame(parser, complete);
    expect_frame(parser, complete);
  }, passed);

  run("bad checksum then valid", [&]() {
    RxStreamParser parser;
    auto bad = make_frame(10);
    bad[22] ^= 0x01;
    const auto good = make_frame(11);
    std::vector<uint8_t> bytes(bad.begin(), bad.end());
    bytes.insert(bytes.end(), good.begin(), good.end());
    parser.append(bytes.data(), bytes.size());
    expect_frame(parser, good);
    expect(parser.stats().checksum_failures == 1, "checksum failure missing");
  }, passed);

  run("bad tail then valid", [&]() {
    RxStreamParser parser;
    auto bad = make_frame(12);
    bad[23] = 0x00;
    const auto good = make_frame(13);
    std::vector<uint8_t> bytes(bad.begin(), bad.end());
    bytes.insert(bytes.end(), good.begin(), good.end());
    parser.append(bytes.data(), bytes.size());
    expect_frame(parser, good);
    expect(parser.stats().tail_failures >= 1, "tail failure missing");
  }, passed);

  run("false header inside payload", [&]() {
    RxStreamParser parser;
    auto bad = make_frame(14);
    bad[4] = origincar_base_rx::kFrameHeader;
    bad[23] = 0x00;
    const auto good = make_frame(15);
    std::vector<uint8_t> bytes(bad.begin(), bad.end());
    bytes.insert(bytes.end(), good.begin(), good.end());
    parser.append(bytes.data(), bytes.size());
    expect_frame(parser, good);
  }, passed);

  run("single append with many frames", [&]() {
    RxStreamParser parser;
    std::vector<Frame> frames;
    std::vector<uint8_t> bytes;
    for (uint8_t seed = 16; seed < 26; ++seed) {
      frames.push_back(make_frame(seed));
      bytes.insert(bytes.end(), frames.back().begin(), frames.back().end());
    }
    parser.append(bytes.data(), bytes.size());
    for (const auto & frame : frames) {
      expect_frame(parser, frame);
    }
  }, passed);

  run("incomplete final frame waits", [&]() {
    RxStreamParser parser;
    const auto first = make_frame(26);
    const auto second = make_frame(27);
    append_frame(parser, first);
    parser.append(second.data(), 19);
    expect_frame(parser, first);
    expect(!parser.has_frame(), "incomplete final frame parsed");
    parser.append(second.data() + 19, second.size() - 19);
    expect_frame(parser, second);
  }, passed);

  run("invalid candidate preserves following frame", [&]() {
    RxStreamParser parser;
    auto bad = make_frame(28);
    bad[22] ^= 0x80;
    const auto good = make_frame(29);
    std::vector<uint8_t> bytes(bad.begin(), bad.end());
    bytes.insert(bytes.end(), good.begin(), good.end());
    parser.append(bytes.data(), bytes.size());
    expect_frame(parser, good);
    expect(parser.stats().discarded_bytes == bad.size(),
      "invalid candidate did not resynchronize byte-by-byte");
  }, passed);

  run("garbage cannot grow buffer", [&]() {
    RxStreamParser parser(64);
    std::vector<uint8_t> garbage(10000, 0x55);
    parser.append(garbage.data(), garbage.size());
    expect(parser.buffer_size() <= 64, "buffer exceeded configured maximum");
    expect(parser.stats().discarded_bytes == garbage.size(), "garbage not discarded");
  }, passed);

  run("one thousand continuous frames", [&]() {
    RxStreamParser parser;
    std::vector<uint8_t> bytes;
    for (int index = 0; index < 1000; ++index) {
      const auto frame = make_frame(static_cast<uint8_t>(index));
      bytes.insert(bytes.end(), frame.begin(), frame.end());
    }
    parser.append(bytes.data(), bytes.size());
    Frame frame{};
    int count = 0;
    while (parser.pop_frame(frame)) {
      ++count;
    }
    expect(count == 1000, "continuous frame count mismatch");
    expect(parser.stats().valid_frames == 1000, "valid frame stats mismatch");
  }, passed);

  run("random garbage between frames", [&]() {
    RxStreamParser parser;
    std::mt19937 generator(12345);
    std::uniform_int_distribution<int> count_distribution(0, 9);
    std::uniform_int_distribution<int> byte_distribution(0, 255);
    std::vector<Frame> frames;
    std::vector<uint8_t> bytes;
    for (int index = 0; index < 200; ++index) {
      for (int count = count_distribution(generator); count > 0; --count) {
        uint8_t value = static_cast<uint8_t>(byte_distribution(generator));
        if (value == origincar_base_rx::kFrameHeader) {
          value = 0x33;
        }
        bytes.push_back(value);
      }
      frames.push_back(make_frame(static_cast<uint8_t>(index + 31)));
      bytes.insert(bytes.end(), frames.back().begin(), frames.back().end());
    }
    parser.append(bytes.data(), bytes.size());
    for (const auto & frame : frames) {
      expect_frame(parser, frame);
    }
    expect(parser.stats().discarded_bytes > 0, "random garbage was not counted");
  }, passed);

  std::cout << "rx parser self-test: " << passed << "/16 passed\n";
  return passed == 16 ? 0 : 1;
}
