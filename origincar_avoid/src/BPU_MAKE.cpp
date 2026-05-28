// 标准C++库
#include <iostream>     // 输入输出流
#include <vector>      // 向量容器
#include <algorithm>   // 算法库
#include <chrono>      // 时间相关功能
#include <iomanip>     // 输入输出格式控制

// OpenCV库
#include <opencv2/opencv.hpp>      // OpenCV主要头文件
#include <opencv2/dnn/dnn.hpp>     // OpenCV深度学习模块

// 地平线RDK BPU API
#include "dnn/hb_dnn.h"           // BPU基础功能
#include "dnn/hb_dnn_ext.h"       // BPU扩展功能
#include "dnn/plugin/hb_dnn_layer.h"    // BPU层定义
#include "dnn/plugin/hb_dnn_plugin.h"   // BPU插件
#include "dnn/hb_sys.h"           // BPU系统功能

// 错误检查宏定义
#define RDK_CHECK_SUCCESS(value, errmsg)                        \
    do                                                          \
    {                                                          \
        auto ret_code = value;                                  \
        if (ret_code != 0)                                      \
        {                                                       \
            std::cout << errmsg << ", error code:" << ret_code; \
            return ret_code;                                    \
        }                                                       \
    } while (0);

// 模型和检测相关的默认参数定义
#define DEFAULT_MODEL_PATH "/root/Deep_Learning/YOLOv5/models/tennis_detect_640x640_bayese_.bin"  // 默认模型路径
#define DEFAULT_CLASSES_NUM 1          // 默认类别数量
#define CLASSES_LIST "tennis_ball"     // 类别名称
#define DEFAULT_NMS_THRESHOLD 0.45f    // 非极大值抑制阈值
#define DEFAULT_SCORE_THRESHOLD 0.25f  // 置信度阈值
#define DEFAULT_NMS_TOP_K 300          // NMS保留的最大框数
#define DEFAULT_FONT_SIZE 1.0f         // 绘制文字大小
#define DEFAULT_FONT_THICKNESS 1.0f    // 绘制文字粗细
#define DEFAULT_LINE_SIZE 2.0f         // 绘制线条粗细

// 运行模式选择
#define DETECT_MODE 1    // 检测模式: 0-单张图片, 1-实时检测
#define ENABLE_DRAW 1    // 绘图开关: 0-禁用, 1-启用
#define LOAD_FROM_DDR 0  // 模型加载方式: 0-从文件加载, 1-从内存加载

// 特征图尺度定义 (基于输入尺寸的倍数关系)
#define H_8 (input_h_ / 8)    // 输入高度的1/8
#define W_8 (input_w_ / 8)    // 输入宽度的1/8
#define H_16 (input_h_ / 16)  // 输入高度的1/16
#define W_16 (input_w_ / 16)  // 输入宽度的1/16
#define H_32 (input_h_ / 32)  // 输入高度的1/32
#define W_32 (input_w_ / 32)  // 输入宽度的1/32

// BPU目标检测类
class BPU_Detect {
public:
    // 构造函数：初始化检测器的参数
    // @param model_path: 模型文件路径
    // @param classes_num: 检测类别数量
    // @param nms_threshold: NMS阈值
    // @param score_threshold: 置信度阈值
    // @param nms_top_k: NMS保留的最大框数
    BPU_Detect(const std::string& model_path = DEFAULT_MODEL_PATH,
                 int classes_num = DEFAULT_CLASSES_NUM,
                 float nms_threshold = DEFAULT_NMS_THRESHOLD,
                 float score_threshold = DEFAULT_SCORE_THRESHOLD,
                 int nms_top_k = DEFAULT_NMS_TOP_K);
    
    // 析构函数：释放资源
    ~BPU_Detect();

    // 主要功能接口
    bool Init();  // 初始化BPU和模型
    bool Detect(const cv::Mat& input_img, cv::Mat& output_img);  // 执行目标检测
    bool Release();  // 释放所有资源

private:
    // 内部工具函数
    bool LoadModel();  // 加载模型文件
    bool GetModelInfo();  // 获取模型的输入输出信息
    bool PreProcess(const cv::Mat& input_img);  // 图像预处理（resize和格式转换）
    bool Inference();  // 执行模型推理
    bool PostProcess();  // 后处理（NMS等）
    void DrawResults(cv::Mat& img);  // 在图像上绘制检测结果
    void PrintResults() const;  // 打印检测结果到控制台

    // 特征图处理辅助函数
    // @param output_tensor: 输出tensor
    // @param height, width: 特征图尺寸
    // @param anchors: 对应尺度的anchor boxes
    // @param conf_thres_raw: 原始置信度阈值
    void ProcessFeatureMap(hbDNNTensor& output_tensor, 
                          int height, int width,
                          const std::vector<std::pair<double, double>>& anchors,
                          float conf_thres_raw);

    // 成员变量（按照构造函数初始化顺序排列）
    std::string model_path_;      // 模型文件路径
    int classes_num_;             // 类别数量
    float nms_threshold_;         // NMS阈值
    float score_threshold_;       // 置信度阈值
    int nms_top_k_;              // NMS保留的最大框数
    bool is_initialized_;         // 初始化状态标志
    float font_size_;            // 绘制文字大小
    float font_thickness_;       // 绘制文字粗细
    float line_size_;            // 绘制线条粗细
    
    // BPU相关变量
    hbPackedDNNHandle_t packed_dnn_handle_;  // 打包模型句柄
    hbDNNHandle_t dnn_handle_;               // 模型句柄
    const char* model_name_;                 // 模型名称
    
    // 输入输出张量
    hbDNNTensor input_tensor_;               // 输入tensor
    hbDNNTensor* output_tensors_;            // 输出tensor数组
    hbDNNTensorProperties input_properties_; // 输入tensor属性
    
    // 任务相关
    hbDNNTaskHandle_t task_handle_;          // 推理任务句柄
    
    // 模型输入参数
    int input_h_;                            // 输入高度
    int input_w_;                            // 输入宽度
    
    // 检测结果存储
    std::vector<std::vector<cv::Rect2d>> bboxes_;  // 每个类别的边界框
    std::vector<std::vector<float>> scores_;       // 每个类别的得分
    std::vector<std::vector<int>> indices_;        // NMS后的索引
    
    // 图像处理参数
    float x_scale_;                          // X方向缩放比例
    float y_scale_;                          // Y方向缩放比例
    int x_shift_;                            // X方向偏移量
    int y_shift_;                            // Y方向偏移量
    cv::Mat resized_img_;                    // 缩放后的图像
    
    // YOLOv5 anchors信息
    std::vector<std::pair<double, double>> s_anchors_;  // 小目标anchors
    std::vector<std::pair<double, double>> m_anchors_;  // 中目标anchors
    std::vector<std::pair<double, double>> l_anchors_;  // 大目标anchors
    
    // 输出处理
    int output_order_[3];                    // 输出顺序映射
    std::vector<std::string> class_names_;   // 类别名称列表
};

// 构造函数实现
BPU_Detect::BPU_Detect(const std::string& model_path,
                          int classes_num,
                          float nms_threshold,
                          float score_threshold,
                          int nms_top_k)
    : model_path_(model_path),
      classes_num_(classes_num),
      nms_threshold_(nms_threshold),
      score_threshold_(score_threshold),
      nms_top_k_(nms_top_k),
      is_initialized_(false),
      font_size_(DEFAULT_FONT_SIZE),
      font_thickness_(DEFAULT_FONT_THICKNESS),
      line_size_(DEFAULT_LINE_SIZE),
      task_handle_(nullptr) 
      {
    
    // 初始化类别名称
    class_names_ = {CLASSES_LIST};
    
    // 初始化anchors
    std::vector<float> anchors = {10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 
                                 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 
                                 116.0, 90.0, 156.0, 198.0, 373.0, 326.0};
    
    // 设置small, medium, large anchors
    for(int i = 0; i < 3; i++) {
        s_anchors_.push_back({anchors[i*2], anchors[i*2+1]});
        m_anchors_.push_back({anchors[i*2+6], anchors[i*2+7]});
        l_anchors_.push_back({anchors[i*2+12], anchors[i*2+13]});
    }
}

// 析构函数实现
BPU_Detect::~BPU_Detect() {
    if(is_initialized_) {
        Release();
    }
}

// 初始化函数实现
bool BPU_Detect::Init() {
    if(is_initialized_) {
        std::cout << "Already initialized!" << std::endl;
        return true;
    }
    
    auto init_start = std::chrono::high_resolution_clock::now();
    
    if(!LoadModel()) {
        std::cout << "Failed to load model!" << std::endl;
        return false;
    }
    
    if(!GetModelInfo()) {
        std::cout << "Failed to get model info!" << std::endl;
        return false;
    }
    
    is_initialized_ = true;
    
    auto init_end = std::chrono::high_resolution_clock::now();
    float init_time = std::chrono::duration_cast<std::chrono::microseconds>(init_end - init_start).count() / 1000.0f;
    
    std::cout << "\n============ Model Loading Time ============" << std::endl;
    std::cout << "Total init time: " << std::fixed << std::setprecision(2) << init_time << " ms" << std::endl;
    std::cout << "=========================================\n" << std::endl;
    
    return true;
}

// 加载模型实现
bool BPU_Detect::LoadModel() {
    // 记录总加载时间的起点
    auto load_start = std::chrono::high_resolution_clock::now();

#if LOAD_FROM_DDR
    // 用于记录从文件读取模型数据的时间
    float read_time = 0.0f;
#endif
    // 用于记录模型初始化的时间
    float init_time = 0.0f;
    
#if LOAD_FROM_DDR
    // =============== 从文件读取模型到内存 ===============
    auto read_start = std::chrono::high_resolution_clock::now();
    
    // 打开模型文件
    FILE* fp = fopen(model_path_.c_str(), "rb");
    if (!fp) {
        std::cout << "Failed to open model file: " << model_path_ << std::endl;
        return false;
    }
    
    // 获取文件大小:
    fseek(fp, 0, SEEK_END);// 1. 将文件指针移到末尾
    size_t model_size = static_cast<size_t>(ftell(fp));// 2. 获取当前位置(即文件大小)
    fseek(fp, 0, SEEK_SET);// 3. 将文件指针重置到开头
    
    // 为模型数据分配内存
    void* model_data = malloc(model_size);
    if (!model_data) {
        std::cout << "Failed to allocate memory for model data" << std::endl;
        fclose(fp);
        return false;
    }
    
    // 读取模型数据到内存
    size_t read_size = fread(model_data, 1, model_size, fp);
    fclose(fp);
    
    // 计算文件读取时间
    auto read_end = std::chrono::high_resolution_clock::now();
    read_time = std::chrono::duration_cast<std::chrono::microseconds>(read_end - read_start).count() / 1000.0f;
    
    // 验证是否完整读取了文件
    if (read_size != model_size) {
        std::cout << "Failed to read model data, expected " << model_size 
                 << " bytes, but got " << read_size << " bytes" << std::endl;
        free(model_data);
        return false;
    }
    
    // =============== 从内存初始化模型 ===============
    auto init_start = std::chrono::high_resolution_clock::now();
    
    // 准备模型数据数组和长度数组
    const void* model_data_array[] = {model_data};
    int32_t model_data_length[] = {static_cast<int32_t>(model_size)};
    
    // 使用BPU API从内存初始化模型
    RDK_CHECK_SUCCESS(
        hbDNNInitializeFromDDR(&packed_dnn_handle_, model_data_array, model_data_length, 1),
        "Initialize model from DDR failed");
    
    // 释放临时分配的内存
    free(model_data);
    
    // 计算模型初始化时间
    auto init_end = std::chrono::high_resolution_clock::now();
    init_time = std::chrono::duration_cast<std::chrono::microseconds>(init_end - init_start).count() / 1000.0f;
    
#else
    // =============== 直接从文件初始化模型 ===============
    auto init_start = std::chrono::high_resolution_clock::now();
    
    // 获取模型文件路径
    const char* model_file_name = model_path_.c_str();
    
    // 使用BPU API从文件初始化模型
    RDK_CHECK_SUCCESS(
        hbDNNInitializeFromFiles(&packed_dnn_handle_, &model_file_name, 1),
        "Initialize model from file failed");
    
    // 计算模型初始化时间
    auto init_end = std::chrono::high_resolution_clock::now();
    init_time = std::chrono::duration_cast<std::chrono::microseconds>(init_end - init_start).count() / 1000.0f;
#endif

    // =============== 计算并打印总时间统计 ===============
    auto load_end = std::chrono::high_resolution_clock::now();
    float total_load_time = std::chrono::duration_cast<std::chrono::microseconds>(load_end - load_start).count() / 1000.0f;

    // 打印时间统计信息
    std::cout << "\n============ Model Loading Details ============" << std::endl;
#if LOAD_FROM_DDR
    std::cout << "File reading time: " << std::fixed << std::setprecision(2) << read_time << " ms" << std::endl;
#endif
    std::cout << "Model init time: " << std::fixed << std::setprecision(2) << init_time << " ms" << std::endl;
    std::cout << "Total loading time: " << std::fixed << std::setprecision(2) << total_load_time << " ms" << std::endl;
    std::cout << "===========================================\n" << std::endl;

    return true;
}

// 获取模型信息实现
bool BPU_Detect::GetModelInfo() {
    // 获取模型名称列表
    const char** model_name_list;
    int model_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle_),
        "hbDNNGetModelNameList failed");
    if(model_count > 1) {
        std::cout << "Model count: " << model_count << std::endl;
        std::cout << "Please check the model count!" << std::endl;
        return false;
    }
    model_name_ = model_name_list[0];
    
    // 获取模型句柄
    RDK_CHECK_SUCCESS(
        hbDNNGetModelHandle(&dnn_handle_, packed_dnn_handle_, model_name_),
        "hbDNNGetModelHandle failed");
    
    // 获取输入信息
    int32_t input_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetInputCount(&input_count, dnn_handle_),
        "hbDNNGetInputCount failed");
    RDK_CHECK_SUCCESS(
        hbDNNGetInputTensorProperties(&input_properties_, dnn_handle_, 0),
        "hbDNNGetInputTensorProperties failed");

    if(input_count > 1){
        std::cout << "模型输入节点大于1，请检查！" << std::endl;
        return false;
    }
    if(input_properties_.validShape.numDimensions == 4){
        std::cout << "输入tensor类型: HB_DNN_IMG_TYPE_NV12" << std::endl;
    }
    else{
        std::cout << "输入tensor类型不是HB_DNN_IMG_TYPE_NV12，请检查！" << std::endl;
        return false;
    }
    if(input_properties_.tensorType == 1){
        std::cout << "输入tensor数据排布: HB_DNN_LAYOUT_NCHW" << std::endl;
    }
    else{
        std::cout << "输入tensor数据排布不是HB_DNN_LAYOUT_NCHW，请检查！" << std::endl;
        return false;
    }
    // 获取输入尺寸
    input_h_ = input_properties_.validShape.dimensionSize[2];
    input_w_ = input_properties_.validShape.dimensionSize[3];
    if (input_properties_.validShape.numDimensions == 4)
    {
        std::cout << "输入的尺寸为: (" << input_properties_.validShape.dimensionSize[0];
        std::cout << ", " << input_properties_.validShape.dimensionSize[1];
        std::cout << ", " << input_h_;
        std::cout << ", " << input_w_ << ")" << std::endl;
    }
    else
    {
        std::cout << "输入的尺寸不是(1,3,640,640)，请检查！" << std::endl;
        return false;
    }
    
    // 获取输出信息并调整输出顺序
    int32_t output_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetOutputCount(&output_count, dnn_handle_),
        "hbDNNGetOutputCount failed");
    
    // 分配输出tensor内存
    output_tensors_ = new hbDNNTensor[output_count];
    memset(output tensors_,0,sizeof(hbDNNTensor)* output count); 
    
    // =============== 调整输出头顺序映射 ===============
    // YOLOv5有3个输出头，分别对应3种不同尺度的特征图
    // 需要确保输出顺序为: 小目标(8倍下采样) -> 中目标(16倍下采样) -> 大目标(32倍下采样)
    
    // 初始化默认顺序
    output_order_[0] = 0;  // 默认第1个输出
    output_order_[1] = 1;  // 默认第2个输出
    output_order_[2] = 2;  // 默认第3个输出

    // 定义期望的输出特征图尺寸和通道数
    int32_t expected_shapes[3][3] = {
        {H_8,  W_8,  3 * (5 + classes_num_)},   // 小目标特征图: H/8 x W/8
        {H_16, W_16, 3 * (5 + classes_num_)},   // 中目标特征图: H/16 x W/16
        {H_32, W_32, 3 * (5 + classes_num_)}    // 大目标特征图: H/32 x W/32
    };

    // 遍历每个期望的输出尺度
    for(int i = 0; i < 3; i++) {
        // 遍历实际的输出节点
        for(int j = 0; j < 3; j++) {
            // 获取当前输出节点的属性
            hbDNNTensorProperties output_properties;
            RDK_CHECK_SUCCESS(
                hbDNNGetOutputTensorProperties(&output_properties, dnn_handle_, j),
                "Get output tensor properties failed");
            
            // 获取实际的特征图尺寸和通道数
            int32_t actual_h = output_properties.validShape.dimensionSize[1];
            int32_t actual_w = output_properties.validShape.dimensionSize[2];
            int32_t actual_c = output_properties.validShape.dimensionSize[3];

            // 如果实际尺寸和通道数与期望的匹配
            if(actual_h == expected_shapes[i][0] && 
               actual_w == expected_shapes[i][1] && 
               actual_c == expected_shapes[i][2]) {
                // 记录正确的输出顺序
                output_order_[i] = j;
                break;
            }
        }
    }

    // 打印输出顺序映射信息
    std::cout << "\n============ Output Order Mapping ============" << std::endl;
    std::cout << "Small object  (1/" << 8  << "): output[" << output_order_[0] << "]" << std::endl;
    std::cout << "Medium object (1/" << 16 << "): output[" << output_order_[1] << "]" << std::endl;
    std::cout << "Large object  (1/" << 32 << "): output[" << output_order_[2] << "]" << std::endl;
    std::cout << "==========================================\n" << std::endl;

    return true;
}

// 检测函数实现
bool BPU_Detect::Detect(const cv::Mat& input_img, cv::Mat& output_img) {
    if(!is_initialized_) {
        std::cout << "Please initialize first!" << std::endl;
        return false;
    }
    
    // 定义所有时间变量
    float preprocess_time = 0.0f;
    float infer_time = 0.0f;
    float postprocess_time = 0.0f;
    float draw_time = 0.0f;
    float total_time = 0.0f;
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
#if ENABLE_DRAW
    input_img.copyTo(output_img);
#endif

    bool success = true;
    
    // 预处理
    {
        auto preprocess_start = std::chrono::high_resolution_clock::now();
        success = PreProcess(input_img);
        auto preprocess_end = std::chrono::high_resolution_clock::now();
        preprocess_time = std::chrono::duration_cast<std::chrono::microseconds>(
            preprocess_end - preprocess_start).count() / 1000.0f;
        
        if (!success) {
            std::cout << "Preprocess failed" << std::endl;
            goto cleanup;  
        }
    }
    
    // 推理
    {
        auto infer_start = std::chrono::high_resolution_clock::now();
        success = Inference();
        auto infer_end = std::chrono::high_resolution_clock::now();
        infer_time = std::chrono::duration_cast<std::chrono::microseconds>(
            infer_end - infer_start).count() / 1000.0f;
        
        if (!success) {
            std::cout << "Inference failed" << std::endl;
            goto cleanup;
        }
    }
    
    // 后处理
    {
        auto postprocess_start = std::chrono::high_resolution_clock::now();
        success = PostProcess();
        auto postprocess_end = std::chrono::high_resolution_clock::now();
        postprocess_time = std::chrono::duration_cast<std::chrono::microseconds>(
            postprocess_end - postprocess_start).count() / 1000.0f;
        
        if (!success) {
            std::cout << "Postprocess failed" << std::endl;
            goto cleanup;
        }
    }
    
    // 绘制结果
    {
        auto draw_start = std::chrono::high_resolution_clock::now();
        DrawResults(output_img);
        auto draw_end = std::chrono::high_resolution_clock::now();
        draw_time = std::chrono::duration_cast<std::chrono::microseconds>(
            draw_end - draw_start).count() / 1000.0f;
    }
    
    // 计算总时间
    {
        auto total_end = std::chrono::high_resolution_clock::now();
        total_time = std::chrono::duration_cast<std::chrono::microseconds>(
            total_end - total_start).count() / 1000.0f;
    }
    
    // 打印时间统计
    std::cout << "\n============ Time Statistics ============" << std::endl;
    std::cout << "Preprocess time: " << std::fixed << std::setprecision(2) << preprocess_time << " ms" << std::endl;
    std::cout << "Inference time: " << std::fixed << std::setprecision(2) << infer_time << " ms" << std::endl;
    std::cout << "Postprocess time: " << std::fixed << std::setprecision(2) << postprocess_time << " ms" << std::endl;
    std::cout << "Draw time: " << std::fixed << std::setprecision(2) << draw_time << " ms" << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(2) << total_time << " ms" << std::endl;
    std::cout << "FPS: " << std::fixed << std::setprecision(2) << 1000.0f / total_time << std::endl;
    std::cout << "======================================\n" << std::endl;

cleanup:
    // 清理资源
    if (task_handle_) {
        hbDNNReleaseTask(task_handle_);
        task_handle_ = nullptr;
    }
    
    // 释放输入内存
    if(input_tensor_.sysMem[0].virAddr) {
        hbSysFreeMem(&(input_tensor_.sysMem[0]));
        input_tensor_.sysMem[0].virAddr = nullptr;
    }
    
    return success;
}

// 预处理实现
bool BPU_Detect::PreProcess(const cv::Mat& input_img) {
    // 使用letterbox方式进行预处理
    x_scale_ = std::min(1.0f * input_h_ / input_img.rows, 1.0f * input_w_ / input_img.cols);
    y_scale_ = x_scale_;
    
    int new_w = input_img.cols * x_scale_;
    x_shift_ = (input_w_ - new_w) / 2;
    int x_other = input_w_ - new_w - x_shift_;
    
    int new_h = input_img.rows * y_scale_;
    y_shift_ = (input_h_ - new_h) / 2;
    int y_other = input_h_ - new_h - y_shift_;
    
    cv::resize(input_img, resized_img_, cv::Size(new_w, new_h));
    cv::copyMakeBorder(resized_img_, resized_img_, y_shift_, y_other, 
                       x_shift_, x_other, cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));
    
    // 转换为NV12格式
    cv::Mat yuv_mat;
    cv::cvtColor(resized_img_, yuv_mat, cv::COLOR_BGR2YUV_I420);
    
    // 准备输入tensor
    hbSysAllocCachedMem(&input_tensor_.sysMem[0], int(3 * input_h_ * input_w_ / 2));
    uint8_t* yuv = yuv_mat.ptr<uint8_t>();
    uint8_t* ynv12 = (uint8_t*)input_tensor_.sysMem[0].virAddr;
    // 计算UV部分的高度和宽度，以及Y部分的大小
    int uv_height = input_h_ / 2;
    int uv_width = input_w_ / 2;
    int y_size = input_h_ * input_w_;
    // 将Y分量数据复制到输入张量
    memcpy(ynv12, yuv, y_size);
    // 获取NV12格式的UV分量位置
    uint8_t* nv12 = ynv12 + y_size;
    uint8_t* u_data = yuv + y_size;
    uint8_t* v_data = u_data + uv_height * uv_width;
    // 将U和V分量交替写入NV12格式
    for(int i = 0; i < uv_width * uv_height; i++) {
        *nv12++ = *u_data++;
        *nv12++ = *v_data++;
    }
    // 将内存缓存清理，确保数据准备好可以供模型使用
    hbSysFlushMem(&input_tensor_.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);// 清除缓存，确保数据同步
    return true;
}

// 推理实现
bool BPU_Detect::Inference() {
    // 确保先释放之前的任务
    if (task_handle_) {
        hbDNNReleaseTask(task_handle_);
        task_handle_ = nullptr;
    }
    
    // 释放之前的输出内存
    for(int i = 0; i < 3; i++) {
        if(output_tensors_ && output_tensors_[i].sysMem[0].virAddr) {
            hbSysFreeMem(&(output_tensors_[i].sysMem[0]));
            output_tensors_[i].sysMem[0].virAddr = nullptr;
        }
    }
    
    // 初始化输入tensor属性
    input_tensor_.properties = input_properties_;
    
    // 获取输出tensor属性并分配内存
    for(int i = 0; i < 3; i++) {
        hbDNNTensorProperties output_properties;
        RDK_CHECK_SUCCESS(
            hbDNNGetOutputTensorProperties(&output_properties, dnn_handle_, i),
            "Get output tensor properties failed");
        output_tensors_[i].properties = output_properties;
        
        // 为输出分配内存
        int out_aligned_size = output_properties.alignedByteSize;
        RDK_CHECK_SUCCESS(
            hbSysAllocCachedMem(&output_tensors_[i].sysMem[0], out_aligned_size),
            "Allocate output memory failed");
    }
    
    // 设置推理控制参数
    hbDNNInferCtrlParam infer_ctrl_param;
    HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
    
    // 执行推理
    RDK_CHECK_SUCCESS(
        hbDNNInfer(&task_handle_, &output_tensors_, &input_tensor_, dnn_handle_, &infer_ctrl_param),
        "Model inference failed");
    
    // 等待任务完成
    RDK_CHECK_SUCCESS(
        hbDNNWaitTaskDone(task_handle_, 0),
        "Wait task done failed");
    
    return true;
}

// 后处理实现
bool BPU_Detect::PostProcess() {
    // 清空上次的结果
    bboxes_.clear();
    scores_.clear();
    indices_.clear();
    
    // 调整大小
    bboxes_.resize(classes_num_);
    scores_.resize(classes_num_);
    indices_.resize(classes_num_);
    
    float conf_thres_raw = -log(1 / score_threshold_ - 1);
    
    // 处理三个尺度的输出
    ProcessFeatureMap(output_tensors_[0], H_8, W_8, s_anchors_, conf_thres_raw);
    ProcessFeatureMap(output_tensors_[1], H_16, W_16, m_anchors_, conf_thres_raw);
    ProcessFeatureMap(output_tensors_[2], H_32, W_32, l_anchors_, conf_thres_raw);
    
    // 对每个类别进行NMS
    for(int i = 0; i < classes_num_; i++) {
        cv::dnn::NMSBoxes(bboxes_[i], scores_[i], score_threshold_, 
                         nms_threshold_, indices_[i], 1.f, nms_top_k_);
    }
    
    return true;
}

// 打印检测结果实现
void BPU_Detect::PrintResults() const {
    // 打印检测结果的总体信息
    int total_detections = 0;
    for(int cls_id = 0; cls_id < classes_num_; cls_id++) {
        total_detections += indices_[cls_id].size();
    }
    std::cout << "\n============ Detection Results ============" << std::endl;
    std::cout << "Total detections: " << total_detections << std::endl;
    
    for(int cls_id = 0; cls_id < classes_num_; cls_id++) {
        if(!indices_[cls_id].empty()) {
            std::cout << "\nClass: " << class_names_[cls_id] << std::endl;
            std::cout << "Number of detections: " << indices_[cls_id].size() << std::endl;
            std::cout << "Details:" << std::endl;
            
            for(size_t i = 0; i < indices_[cls_id].size(); i++) {
                int idx = indices_[cls_id][i];
                float x1 = (bboxes_[cls_id][idx].x - x_shift_) / x_scale_;
                float y1 = (bboxes_[cls_id][idx].y - y_shift_) / y_scale_;
                float x2 = x1 + (bboxes_[cls_id][idx].width) / x_scale_;
                float y2 = y1 + (bboxes_[cls_id][idx].height) / y_scale_;
                float score = scores_[cls_id][idx];
                
                // 打印每个检测框的详细信息
                std::cout << "  Detection " << i + 1 << ":" << std::endl;
                std::cout << "    Position: (" << x1 << ", " << y1 << ") to (" << x2 << ", " << y2 << ")" << std::endl;
                std::cout << "    Confidence: " << std::fixed << std::setprecision(2) << score * 100 << "%" << std::endl;
            }
        }
    }
    std::cout << "========================================\n" << std::endl;
}

// 绘制结果实现
void BPU_Detect::DrawResults(cv::Mat& img) {
#if ENABLE_DRAW
    for(int cls_id = 0; cls_id < classes_num_; cls_id++) {
        if(!indices_[cls_id].empty()) {
            for(size_t i = 0; i < indices_[cls_id].size(); i++) {
                int idx = indices_[cls_id][i];
                float x1 = (bboxes_[cls_id][idx].x - x_shift_) / x_scale_;
                float y1 = (bboxes_[cls_id][idx].y - y_shift_) / y_scale_;
                float x2 = x1 + (bboxes_[cls_id][idx].width) / x_scale_;
                float y2 = y1 + (bboxes_[cls_id][idx].height) / y_scale_;
                float score = scores_[cls_id][idx];
                
                // 绘制边界框
                cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), 
                            cv::Scalar(255, 0, 0), line_size_);
                
                // 绘制标签
                std::string text = class_names_[cls_id] + ": " + 
                                std::to_string(static_cast<int>(score * 100)) + "%";
                cv::putText(img, text, cv::Point(x1, y1 - 5), 
                          cv::FONT_HERSHEY_SIMPLEX, font_size_, 
                          cv::Scalar(0, 0, 255), font_thickness_, cv::LINE_AA);
            }
        }
    }
#endif
    // 打印检测结果
    PrintResults();
}

// 特征图处理辅助函数
void BPU_Detect::ProcessFeatureMap(hbDNNTensor& output_tensor, 
                                  int height, int width,
                                  const std::vector<std::pair<double, double>>& anchors,
                                  float conf_thres_raw) {
    // 检查量化类型
    if (output_tensor.properties.quantiType != NONE) {
        std::cout << "Output tensor quantization type should be NONE!" << std::endl;
        return;
    }
    
    // 刷新内存
    hbSysFlushMem(&output_tensor.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    
    // 获取输出数据指针
    auto* raw_data = reinterpret_cast<float*>(output_tensor.sysMem[0].virAddr);
    
    // 遍历特征图的每个位置
    for(int h = 0; h < height; h++) {
        for(int w = 0; w < width; w++) {
            for(const auto& anchor : anchors) {
                // 获取当前位置的预测数据
                float* cur_raw = raw_data;
                raw_data += (5 + classes_num_);
                
                // 条件概率过滤
                if(cur_raw[4] < conf_thres_raw) continue;
                
                // 找到最大类别概率
                int cls_id = 5;
                int end = classes_num_ + 5;
                for(int i = 6; i < end; i++) {
                    if(cur_raw[i] > cur_raw[cls_id]) {
                        cls_id = i;
                    }
                }
                
                // 计算最终得分
                float score = 1.0f / (1.0f + std::exp(-cur_raw[4])) * 
                            1.0f / (1.0f + std::exp(-cur_raw[cls_id]));
                
                // 得分过滤
                if(score < score_threshold_) continue;
                cls_id -= 5;
                
                // 解码边界框
                float stride = input_h_ / height;
                float center_x = ((1.0f / (1.0f + std::exp(-cur_raw[0]))) * 2 - 0.5f + w) * stride;
                float center_y = ((1.0f / (1.0f + std::exp(-cur_raw[1]))) * 2 - 0.5f + h) * stride;
                float bbox_w = std::pow((1.0f / (1.0f + std::exp(-cur_raw[2]))) * 2, 2) * anchor.first;
                float bbox_h = std::pow((1.0f / (1.0f + std::exp(-cur_raw[3]))) * 2, 2) * anchor.second;
                float bbox_x = center_x - bbox_w / 2.0f;
                float bbox_y = center_y - bbox_h / 2.0f;
                
                // 保存检测结果
                bboxes_[cls_id].push_back(cv::Rect2d(bbox_x, bbox_y, bbox_w, bbox_h));
                scores_[cls_id].push_back(score);
            }
        }
    }
}

// 释放资源实现
bool BPU_Detect::Release() {
    if(!is_initialized_) {
        return true;
    }
    
    // 释放任务
    if(task_handle_) {
        hbDNNReleaseTask(task_handle_);
        task_handle_ = nullptr;
    }
    
    try {
        // 释放输入内存
        if(input_tensor_.sysMem[0].virAddr) {
            hbSysFreeMem(&(input_tensor_.sysMem[0]));
        }
        
        // 释放输出内存
        for(int i = 0; i < 3; i++) {
            if(output_tensors_ && output_tensors_[i].sysMem[0].virAddr) {
                hbSysFreeMem(&(output_tensors_[i].sysMem[0]));
            }
        }
        
        if(output_tensors_) {
            delete[] output_tensors_;
            output_tensors_ = nullptr;
        }
        
        // 释放模型
        if(packed_dnn_handle_) {
            hbDNNRelease(packed_dnn_handle_);
            packed_dnn_handle_ = nullptr;
        }
    } catch(const std::exception& e) {
        std::cout << "Exception during release: " << e.what() << std::endl;
    }
    
    is_initialized_ = false;
    return true;
}

// 修改main函数
int main() {
    // 创建检测器实例
    BPU_Detect detector;
    
    // 初始化
    if (!detector.Init()) {
        std::cout << "Failed to initialize detector" << std::endl;
        return -1;
    }

#if DETECT_MODE == 0
    // 单张图片检测模式
    std::cout << "Single image detection mode" << std::endl;
    
    // 读取测试图片
    cv::Mat input_img = cv::imread("/root/Deep_Learning/YOLOv5/imgs/tennis_1_frame_0001.jpg");
    if (input_img.empty()) {
        std::cout << "Failed to load image" << std::endl;
        return -1;
    }
    
    // 执行检测
    cv::Mat output_img;
#if ENABLE_DRAW
    if (!detector.Detect(input_img, output_img)) {
        std::cout << "Detection failed" << std::endl;
        return -1;
    }
    // 保存结果
    cv::imwrite("cpp_result.jpg", output_img);
#else
    if (!detector.Detect(input_img, output_img)) {
        std::cout << "Detection failed" << std::endl;
        return -1;
    }
#endif

#else
    // 实时检测模式
    std::cout << "Real-time detection mode" << std::endl;
    
    // 打开摄像头
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Failed to open camera" << std::endl;
        return -1;
    }
    
    cv::Mat frame, output_frame;
    while (true) {
        // 读取一帧
        cap >> frame;
        if (frame.empty()) {
            std::cout << "Failed to read frame" << std::endl;
            break;
        }
        
        // 执行检测
        if (!detector.Detect(frame, output_frame)) {
            std::cout << "Detection failed" << std::endl;
            break;
        }
        
#if ENABLE_DRAW
        // 显示结果
        cv::imshow("Real-time Detection", output_frame);
        
        // 按'q'退出
        if (cv::waitKey(1) == 'q') {
            break;
        }
#endif
    }
    
#if ENABLE_DRAW
    // 释放摄像头
    cap.release();
    cv::destroyAllWindows();
#endif
#endif
    
    // 释放资源
    detector.Release();
    
    return 0;
}

