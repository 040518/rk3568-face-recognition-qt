#include "v4l2_capture.h"

#include <cerrno>
#include <cstdio>
#include <cstring>

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <opencv2/imgproc.hpp>

V4l2Capture::V4l2Capture() = default;

V4l2Capture::~V4l2Capture()
{
    close();
}

bool V4l2Capture::open(const string& device,
                       unsigned int requested_width,
                       unsigned int requested_height,
                       unsigned int requested_buffer_count,
                       unsigned int requested_fps)
{
    close();

    fd_ = ::open(device.c_str(), O_RDWR);
    if (fd_ < 0) {
        fprintf(stderr,
                "ERROR: open %s failed: %s\n",
                device.c_str(),
                strerror(errno));
        return false;
    }

    device_ = device;

    v4l2_capability capability{};
    if (ioctl(fd_, VIDIOC_QUERYCAP, &capability) < 0) {
        fprintf(stderr,
                "ERROR: VIDIOC_QUERYCAP failed: %s\n",
                strerror(errno));
        close();
        return false;
    }

    unsigned int caps = capability.capabilities;
    if ((capability.capabilities & V4L2_CAP_DEVICE_CAPS) != 0) {
        caps = capability.device_caps;
    }

    printf("driver=%s\n",
           reinterpret_cast<const char*>(capability.driver));
    printf("card=%s\n",
           reinterpret_cast<const char*>(capability.card));

    if ((caps & V4L2_CAP_VIDEO_CAPTURE_MPLANE) == 0) {
        fprintf(stderr,
                "ERROR: device does not support MPlane capture\n");
        close();
        return false;
    }

    if ((caps & V4L2_CAP_STREAMING) == 0) {
        fprintf(stderr,
                "ERROR: device does not support streaming I/O\n");
        close();
        return false;
    }

    v4l2_format format{};
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    format.fmt.pix_mp.width = requested_width;
    format.fmt.pix_mp.height = requested_height;
    format.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
    format.fmt.pix_mp.field = V4L2_FIELD_NONE;

    if (ioctl(fd_, VIDIOC_S_FMT, &format) < 0) {
        fprintf(stderr,
                "ERROR: VIDIOC_S_FMT failed: %s\n",
                strerror(errno));
        close();
        return false;
    }

    if (format.fmt.pix_mp.pixelformat != V4L2_PIX_FMT_NV12) {
        fprintf(stderr,
                "ERROR: driver did not accept NV12 format\n");
        close();
        return false;
    }

    if (format.fmt.pix_mp.num_planes != 1) {
        fprintf(stderr,
                "ERROR: expected one plane, got %u\n",
                format.fmt.pix_mp.num_planes);
        close();
        return false;
    }

    width_ = format.fmt.pix_mp.width;
    height_ = format.fmt.pix_mp.height;
    stride_ = format.fmt.pix_mp.plane_fmt[0].bytesperline;
    plane_count_ = format.fmt.pix_mp.num_planes;

    printf("width=%u\n", width_);
    printf("height=%u\n", height_);
    printf("pixel_format=NV12\n");
    printf("num_planes=%u\n", plane_count_);
    printf("stride=%u\n", stride_);
    printf("sizeimage=%u\n",
           format.fmt.pix_mp.plane_fmt[0].sizeimage);

    if (requested_fps > 0) {
        v4l2_streamparm parameters{};
        parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        parameters.parm.capture.timeperframe.numerator = 1;
        parameters.parm.capture.timeperframe.denominator = requested_fps;
        if (ioctl(fd_, VIDIOC_S_PARM, &parameters) < 0) {
            fprintf(stderr,
                    "WARN: VIDIOC_S_PARM %u FPS failed: %s\n",
                    requested_fps,
                    strerror(errno));
        }
    }

    v4l2_streamparm active_parameters{};
    active_parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (ioctl(fd_, VIDIOC_G_PARM, &active_parameters) == 0 &&
        active_parameters.parm.capture.timeperframe.numerator > 0) {
        fps_ = static_cast<double>(
            active_parameters.parm.capture.timeperframe.denominator) /
            active_parameters.parm.capture.timeperframe.numerator;
        printf("camera_fps_requested=%u camera_fps_active=%.2f\n",
               requested_fps,
               fps_);
    } else {
        printf("camera_fps_requested=%u camera_fps_active=driver_unspecified\n",
               requested_fps);
    }

    v4l2_requestbuffers request{};
    request.count = requested_buffer_count;
    request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    request.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd_, VIDIOC_REQBUFS, &request) < 0) {
        fprintf(stderr,
                "ERROR: VIDIOC_REQBUFS failed: %s\n",
                strerror(errno));
        close();
        return false;
    }

    buffers_allocated_ = true;

    if (request.count == 0) {
        fprintf(stderr,
                "ERROR: driver allocated zero buffers\n");
        close();
        return false;
    }

    mapped_buffers_.resize(request.count);

    for (unsigned int index = 0;
         index < request.count;
         ++index) {
        v4l2_plane planes[VIDEO_MAX_PLANES]{};
        v4l2_buffer buffer{};

        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = index;
        buffer.length = plane_count_;
        buffer.m.planes = planes;

        if (ioctl(fd_, VIDIOC_QUERYBUF, &buffer) < 0) {
            fprintf(stderr,
                    "ERROR: VIDIOC_QUERYBUF index=%u failed: %s\n",
                    index,
                    strerror(errno));
            close();
            return false;
        }

        if (buffer.length != plane_count_ || planes[0].length == 0) {
            fprintf(stderr,
                    "ERROR: invalid buffer description index=%u\n",
                    index);
            close();
            return false;
        }

        void* address = mmap(nullptr,
                             planes[0].length,
                             PROT_READ | PROT_WRITE,
                             MAP_SHARED,
                             fd_,
                             static_cast<off_t>(
                                 planes[0].m.mem_offset));

        if (address == MAP_FAILED) {
            fprintf(stderr,
                    "ERROR: mmap index=%u failed: %s\n",
                    index,
                    strerror(errno));
            close();
            return false;
        }

        mapped_buffers_[index].address = address;
        mapped_buffers_[index].length = planes[0].length;

        printf("buffer[%u].mapped address=%p length=%u\n",
               index,
               address,
               planes[0].length);
    }

    for (unsigned int index = 0;
         index < mapped_buffers_.size();
         ++index) {
        v4l2_plane planes[VIDEO_MAX_PLANES]{};
        v4l2_buffer buffer{};

        buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buffer.memory = V4L2_MEMORY_MMAP;
        buffer.index = index;
        buffer.length = plane_count_;
        buffer.m.planes = planes;

        if (ioctl(fd_, VIDIOC_QBUF, &buffer) < 0) {
            fprintf(stderr,
                    "ERROR: initial VIDIOC_QBUF index=%u failed: %s\n",
                    index,
                    strerror(errno));
            close();
            return false;
        }
    }

    enum v4l2_buf_type stream_type =
        V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;

    if (ioctl(fd_, VIDIOC_STREAMON, &stream_type) < 0) {
        fprintf(stderr,
                "ERROR: VIDIOC_STREAMON failed: %s\n",
                strerror(errno));
        close();
        return false;
    }

    streaming_ = true;

    printf("stream_started buffers=%zu\n",
           mapped_buffers_.size());

    return true;
}

bool V4l2Capture::read(cv::Mat& bgr_frame,
                       V4l2FrameMetadata& metadata)
{
    if (fd_ < 0 || !streaming_) {
        fprintf(stderr,
                "ERROR: camera is not streaming\n");
        return false;
    }

    v4l2_plane planes[VIDEO_MAX_PLANES]{};
    v4l2_buffer buffer{};

    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    buffer.memory = V4L2_MEMORY_MMAP;
    buffer.length = plane_count_;
    buffer.m.planes = planes;

    if (ioctl(fd_, VIDIOC_DQBUF, &buffer) < 0) {
        fprintf(stderr,
                "ERROR: VIDIOC_DQBUF failed: %s\n",
                strerror(errno));
        return false;
    }

    bool frame_ok = true;

    if (buffer.length != plane_count_ ||
        buffer.index >= mapped_buffers_.size()) {
        fprintf(stderr,
                "ERROR: invalid dequeued buffer index=%u planes=%u\n",
                buffer.index,
                buffer.length);
        frame_ok = false;
    }

    metadata.sequence = buffer.sequence;
    metadata.timestamp_seconds =
        static_cast<long>(buffer.timestamp.tv_sec);
    metadata.timestamp_microseconds =
        static_cast<long>(buffer.timestamp.tv_usec);
    metadata.bytes_used = planes[0].bytesused;

    if (frame_ok) {
        const size_t expected_bytes =
            static_cast<size_t>(stride_) *
            static_cast<size_t>(height_) * 3 / 2;

        const MappedBuffer& mapped =
            mapped_buffers_[buffer.index];

        if (metadata.bytes_used < expected_bytes ||
            mapped.length < expected_bytes) {
            fprintf(stderr,
                    "ERROR: NV12 buffer too small, "
                    "expected=%zu bytesused=%zu mapped=%zu\n",
                    expected_bytes,
                    metadata.bytes_used,
                    mapped.length);
            frame_ok = false;
        } else {
            try {
                cv::Mat nv12_view(
                    static_cast<int>(height_ * 3 / 2),
                    static_cast<int>(width_),
                    CV_8UC1,
                    mapped.address,
                    stride_);

                cv::cvtColor(nv12_view,
                             bgr_frame,
                             cv::COLOR_YUV2BGR_NV12);
            } catch (const cv::Exception& error) {
                fprintf(stderr,
                        "ERROR: NV12 to BGR failed: %s\n",
                        error.what());
                frame_ok = false;
            }
        }
    }

    if (ioctl(fd_, VIDIOC_QBUF, &buffer) < 0) {
        fprintf(stderr,
                "ERROR: VIDIOC_QBUF index=%u failed: %s\n",
                buffer.index,
                strerror(errno));
        return false;
    }

    return frame_ok;
}

void V4l2Capture::close()
{
    if (fd_ < 0) {
        return;
    }

    if (streaming_) {
        enum v4l2_buf_type stream_type =
            V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;

        if (ioctl(fd_, VIDIOC_STREAMOFF, &stream_type) < 0) {
            fprintf(stderr,
                    "ERROR: VIDIOC_STREAMOFF failed: %s\n",
                    strerror(errno));
        } else {
            printf("stream_stopped\n");
        }

        streaming_ = false;
    }

    for (size_t index = 0;
         index < mapped_buffers_.size();
         ++index) {
        MappedBuffer& mapped = mapped_buffers_[index];

        if (mapped.address == MAP_FAILED) {
            continue;
        }

        if (munmap(mapped.address, mapped.length) < 0) {
            fprintf(stderr,
                    "ERROR: munmap index=%zu failed: %s\n",
                    index,
                    strerror(errno));
        } else {
            printf("buffer[%zu].unmapped\n", index);
        }

        mapped.address = MAP_FAILED;
        mapped.length = 0;
    }

    mapped_buffers_.clear();

    if (buffers_allocated_) {
        v4l2_requestbuffers release{};
        release.count = 0;
        release.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        release.memory = V4L2_MEMORY_MMAP;

        if (ioctl(fd_, VIDIOC_REQBUFS, &release) < 0) {
            fprintf(stderr,
                    "ERROR: release buffers failed: %s\n",
                    strerror(errno));
        } else {
            printf("buffers_released\n");
        }

        buffers_allocated_ = false;
    }

    if (::close(fd_) < 0) {
        fprintf(stderr,
                "ERROR: close %s failed: %s\n",
                device_.c_str(),
                strerror(errno));
    } else {
        printf("device_closed\n");
    }

    fd_ = -1;
    device_.clear();
    width_ = 0;
    height_ = 0;
    stride_ = 0;
    fps_ = 0.0;
    plane_count_ = 0;
}

bool V4l2Capture::is_opened() const
{
    return fd_ >= 0 && streaming_;
}

unsigned int V4l2Capture::width() const
{
    return width_;
}

unsigned int V4l2Capture::height() const
{
    return height_;
}

unsigned int V4l2Capture::stride() const
{
    return stride_;
}

double V4l2Capture::fps() const
{
    return fps_;
}

unsigned int V4l2Capture::buffer_count() const
{
    return static_cast<unsigned int>(mapped_buffers_.size());
}
