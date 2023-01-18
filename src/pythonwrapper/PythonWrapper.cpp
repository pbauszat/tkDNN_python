//
// A Python Wrapper around the tkdnn detection network.
//
#include <iostream>
#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "opencv2/opencv.hpp"

#include "CenternetDetection.h"
#include "MobilenetDetection.h"
#include "Yolo3Detection.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Auxilliary Types
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct BoundingBox {
    float x{ 0.0f };
    float y{ 0.0f };
    float width{ 0.0f };
    float height{ 0.0f };
};

struct Detection {
    int class_id{ -1 };
    std::string class_name{};
    BoundingBox box{};
    float probability{ 0.0f };
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Object Detector
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class ObjectDetector {
    typedef py::array_t<uint8_t, py::array::c_style | py::array::forcecast> Image;

public:
    enum Type {
        Yolo = 0,
        CenterNet = 1,
        MobileNet = 2,
    };

public:
    ObjectDetector(std::string const& network_rt_filename, int class_count = 80, float confidence_threshold = 0.3f, int max_batch_size = 1, Type type = Type::Yolo)
        : _network_rt_filename(network_rt_filename), _class_count(class_count), _confidence_threshold(confidence_threshold), _max_batch_size(max_batch_size), _type(type) {
        // Create the internal detection network dependeing on the type
        int fixed_class_count = class_count;
        if (type == Type::CenterNet) {
            _network = std::make_unique<tk::dnn::CenternetDetection>();
        }
        else if (type == Type::MobileNet) {
            _network = std::make_unique<tk::dnn::MobilenetDetection>();
            fixed_class_count++;
        }
        else {
            _network = std::make_unique<tk::dnn::Yolo3Detection>();
        }
        // Initialize the network
        _network->init(network_rt_filename, fixed_class_count, max_batch_size, confidence_threshold);
        std::cout << "Network initialization successful.\n";
    }

    /*
     *todo: improve copying and type conversion, e.g., use python types or custom conversion [https://pybind11.readthedocs.io/en/latest/advanced/cast/overview.html]
     */
    std::vector<std::vector<Detection>> infer(std::vector<Image> images) {
        // If more images are provided than the maximum batch sizr would allow, simply return empty detections
        // todo: raise a proper exception here.
        if (images.size() > _max_batch_size) {
            std::cerr << "More images were given than the maximum batch size allows.\n";
            return {};
        }

        // Convert the numpy array input images to cv::Mat.
        // todo: it seems to be working safely with the cv.Mat numpy arrays, but is this generally true?
        std::vector<cv::Mat> frames;
        for (auto const& image : images) {
            auto const rows = image.shape(0);
            auto const cols = image.shape(1);
            auto const type = CV_8UC3;
            cv::Mat mat(rows, cols, type, (unsigned char*)image.data());
            frames.push_back(mat);
        }

        // Perform inference
        int image_count = static_cast<int>(frames.size());
        _network->update(frames, image_count);

        // Convert detections
        std::vector<std::vector<Detection>> detections;
        for (int i = 0; i < image_count; ++i) {
            std::vector<Detection> image_detections;
            for (auto const& network_detection : _network->batchDetected[i]) {
                Detection detection;
                detection.class_id = network_detection.cl;
                detection.class_name = _network->classesNames[network_detection.cl];
                detection.box = BoundingBox{ network_detection.x, network_detection.y, network_detection.w, network_detection.h };
                detection.probability = network_detection.prob;
                image_detections.push_back(detection);
            }
            detections.push_back(image_detections);
        }
        return detections;
    }

private:
    std::string _network_rt_filename{};
    int _class_count{ 80 };
    float _confidence_threshold{ 0.3f };
    int _max_batch_size{ 1 };
    Type _type{ Type::Yolo };
    std::unique_ptr<tk::dnn::DetectionNN> _network;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Bindings
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(pythonwrapper, m) {
    m.doc() = "Python wrappings for tkdnn.";

    py::class_<BoundingBox>(m, "BoundingBox", py::dynamic_attr())
        .def(py::init<>())
        .def_readwrite("x", &BoundingBox::x)
        .def_readwrite("y", &BoundingBox::y)
        .def_readwrite("width", &BoundingBox::width)
        .def_readwrite("height", &BoundingBox::height)
        .def("__repr__",
            [](const BoundingBox& a) {
                return "<pythonwrapper.BoundingBox (" + std::to_string(a.x) + ", " + std::to_string(a.y) + ", " + std::to_string(a.width) + ", " + std::to_string(a.height) + ")>";
            }
        );

    py::class_<Detection>(m, "Detection", py::dynamic_attr())
        .def(py::init<>())
        .def_readwrite("class_id", &Detection::class_id)
        .def_readwrite("class_name", &Detection::class_name)
        .def_readwrite("box", &Detection::box)
        .def_readwrite("probability", &Detection::probability)
        .def("__repr__",
            [](const Detection& a) {
                return "<pythonwrapper.Detection (" + a.class_name + " [" + std::to_string(a.class_id) + "])>";
            }
        );

    py::class_<ObjectDetector> object_detector(m, "ObjectDetector");

    py::enum_<ObjectDetector::Type>(object_detector, "Type")
        .value("Yolo", ObjectDetector::Type::Yolo)
        .value("CenterNet", ObjectDetector::Type::CenterNet)
        .value("MobileNet", ObjectDetector::Type::MobileNet)
        .export_values();

    object_detector
        .def(py::init<std::string const&, int, float, int, ObjectDetector::Type>(), py::arg("network_rt_filename"), py::arg("class_count") = 80, py::arg("confidence_threshold") = 0.3f, py::arg("max_batch_size") = 1, py::arg("type") = ObjectDetector::Type::Yolo)
        .def("infer", &ObjectDetector::infer, py::arg("images"))
        .def("__repr__",
            [](const ObjectDetector& _) {
                return "<pythonwrapper.ObjectDetector>";
            }
        );

}
