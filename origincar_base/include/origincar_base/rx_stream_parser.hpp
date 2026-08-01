#ifndef ORIGINCAR_BASE_RX_STREAM_PARSER_HPP_
#define ORIGINCAR_BASE_RX_STREAM_PARSER_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

namespace origincar_base_rx
{

constexpr std::size_t kFrameSize = 24;
constexpr uint8_t kFrameHeader = 0x7B;
constexpr uint8_t kFrameTail = 0x7D;

using Frame = std::array<uint8_t, kFrameSize>;

struct RxStats
{
  uint64_t bytes_total = 0;
  uint64_t valid_frames = 0;
  uint64_t checksum_failures = 0;
  uint64_t tail_failures = 0;
  uint64_t resync_events = 0;
  uint64_t discarded_bytes = 0;
  uint64_t overflow_events = 0;
};

class RxStreamParser
{
public:
  explicit RxStreamParser(std::size_t max_buffer_size = 4096)
  : max_buffer_size_(std::max(max_buffer_size, kFrameSize))
  {
    buffer_.reserve(std::min<std::size_t>(max_buffer_size_, 512));
  }

  void append(const uint8_t * data, std::size_t size)
  {
    if (data == nullptr || size == 0) {
      return;
    }
    stats_.bytes_total += size;

    std::size_t offset = 0;
    while (offset < size) {
      if (buffer_.size() == max_buffer_size_) {
        recover_from_overflow();
      }
      const std::size_t capacity = max_buffer_size_ - buffer_.size();
      const std::size_t chunk = std::min(capacity, size - offset);
      buffer_.insert(buffer_.end(), data + offset, data + offset + chunk);
      offset += chunk;
      parse_available();
    }
  }

  bool pop_frame(Frame & frame)
  {
    if (frames_.empty()) {
      return false;
    }
    frame = frames_.front();
    frames_.pop_front();
    return true;
  }

  bool has_frame() const
  {
    return !frames_.empty();
  }

  std::size_t buffer_size() const
  {
    return buffer_.size();
  }

  std::size_t queued_frame_count() const
  {
    return frames_.size();
  }

  const RxStats & stats() const
  {
    return stats_;
  }

private:
  static uint8_t checksum(const uint8_t * data)
  {
    uint8_t value = 0;
    for (std::size_t index = 0; index < 22; ++index) {
      value ^= data[index];
    }
    return value;
  }

  void discard_prefix(std::size_t count)
  {
    if (count == 0) {
      return;
    }
    buffer_.erase(buffer_.begin(), buffer_.begin() + count);
    stats_.discarded_bytes += count;
  }

  void parse_available()
  {
    while (!buffer_.empty()) {
      const auto header = std::find(buffer_.begin(), buffer_.end(), kFrameHeader);
      if (header == buffer_.end()) {
        stats_.resync_events++;
        discard_prefix(buffer_.size());
        return;
      }

      const std::size_t garbage = static_cast<std::size_t>(
        std::distance(buffer_.begin(), header));
      if (garbage > 0) {
        stats_.resync_events++;
        discard_prefix(garbage);
      }

      if (buffer_.size() < kFrameSize) {
        return;
      }

      if (buffer_[kFrameSize - 1] != kFrameTail) {
        stats_.tail_failures++;
        stats_.resync_events++;
        discard_prefix(1);
        continue;
      }

      if (checksum(buffer_.data()) != buffer_[kFrameSize - 2]) {
        stats_.checksum_failures++;
        stats_.resync_events++;
        discard_prefix(1);
        continue;
      }

      Frame frame{};
      std::copy_n(buffer_.begin(), kFrameSize, frame.begin());
      frames_.push_back(frame);
      stats_.valid_frames++;
      buffer_.erase(buffer_.begin(), buffer_.begin() + kFrameSize);
    }
  }

  void recover_from_overflow()
  {
    stats_.overflow_events++;
    stats_.resync_events++;

    const std::size_t keep_limit = kFrameSize - 1;
    const std::size_t search_begin = buffer_.size() > keep_limit ?
      buffer_.size() - keep_limit : 0;
    auto recent_header = std::find(
      buffer_.begin() + search_begin, buffer_.end(), kFrameHeader);
    if (recent_header == buffer_.end()) {
      discard_prefix(buffer_.size());
      return;
    }
    discard_prefix(static_cast<std::size_t>(
      std::distance(buffer_.begin(), recent_header)));
  }

  const std::size_t max_buffer_size_;
  std::vector<uint8_t> buffer_;
  std::deque<Frame> frames_;
  RxStats stats_;
};

}  // namespace origincar_base_rx

#endif  // ORIGINCAR_BASE_RX_STREAM_PARSER_HPP_
