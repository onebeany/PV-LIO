//#include "utility.h" // Assuming this is not strictly needed for the core logic shown
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/io.h> // For pcl_isnan
#include <pcl_conversions/pcl_conversions.h>
#include <string>
#include <vector>
#include <cmath> // For std::fabs, pcl_isnan uses this indirectly

// Global variable to store output type (relevant for RoboSense conversion)
std::string output_type;
// Global variable to control timestamp usage (relevant for RoboSense conversion)
bool use_frame_time;

// --- Ring ID Maps (For RoboSense) ---
static int RING_ID_MAP_RUBY[] = {
        3, 66, 33, 96, 11, 74, 41, 104, 19, 82, 49, 112, 27, 90, 57, 120,
        35, 98, 1, 64, 43, 106, 9, 72, 51, 114, 17, 80, 59, 122, 25, 88,
        67, 34, 97, 0, 75, 42, 105, 8, 83, 50, 113, 16, 91, 58, 121, 24,
        99, 2, 65, 32, 107, 10, 73, 40, 115, 18, 81, 48, 123, 26, 89, 56,
        7, 70, 37, 100, 15, 78, 45, 108, 23, 86, 53, 116, 31, 94, 61, 124,
        39, 102, 5, 68, 47, 110, 13, 76, 55, 118, 21, 84, 63, 126, 29, 92,
        71, 38, 101, 4, 79, 46, 109, 12, 87, 54, 117, 20, 95, 62, 125, 28,
        103, 6, 69, 36, 111, 14, 77, 44, 119, 22, 85, 52, 127, 30, 93, 60
};
static int RING_ID_MAP_16[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 15, 14, 13, 12, 11, 10, 9, 8
};

// --- PCL Point Struct Definitions ---

// RoboSense 포맷 (기존 코드)
struct RsPointXYZIRT {
    PCL_ADD_POINT4D;
    uint8_t intensity;
    uint16_t ring = 0;
    double timestamp = 0;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(RsPointXYZIRT,
                                  (float, x, x)(float, y, y)(float, z, z)(uint8_t, intensity, intensity) // intensity 매핑 수정
                                          (uint16_t, ring, ring)(double, timestamp, timestamp))

// Velodyne 포맷 (기존 코드 - 출력용)
struct VelodynePointXYZIRT {
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY; // float intensity 추가
    uint16_t ring;
    float time; // 상대 시간 (초 단위)

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
                                   (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
                                           (uint16_t, ring, ring)(float, time, time)
)

struct VelodynePointXYZIR {
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY; // float intensity 추가
    uint16_t ring;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIR,
                                   (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
                                           (uint16_t, ring, ring)
)

// ML-X 입력 메시지 포맷 (Workaround 용 - 추가)
// x, y, z (float), rgb (float) 필드를 가짐
struct MLXPointXYZRGBF {
    PCL_ADD_POINT4D; // x, y, z, padding 추가
    float rgb;       // 메시지의 float 'rgb' 필드
    PCL_MAKE_ALIGNED_OPERATOR_NEW // EIGEN 정렬 관련 매크로 추가
} EIGEN_ALIGN16;
// PCL에 새 구조체 등록 (추가)
POINT_CLOUD_REGISTER_POINT_STRUCT (MLXPointXYZRGBF,
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, rgb, rgb) // 'rgb' 필드 매핑
)


// --- ROS Publisher / Subscriber ---
ros::Subscriber subRobosensePC; // 이름 변경 (범용적)
ros::Publisher pubRobosensePC; // 이름 변경 (출력 토픽 명시)

// --- Utility Functions (기존 코드) ---
template<typename T>
bool has_nan(T point) {
    // remove nan point, or the feature assocaion will crash, the surf point will containing nan points
    // pcl remove nan not work normally
    return pcl_isnan(point.x) || pcl_isnan(point.y) || pcl_isnan(point.z);
}

// publish_points 함수: RoboSense 핸들러에서 사용 (기존 코드 수정)
// 수정된 publish_points 함수 시그니처
// PointCloudPtrT는 pcl::PointCloud<...>::Ptr 타입을 직접 받도록 변경
template<typename PointCloudPtrT>
void publish_points(const PointCloudPtrT &new_pc, // Point Cloud Ptr 타입을 직접 받음
                    const sensor_msgs::PointCloud2 &old_msg,
                    double update_timestamp = 0.0) // double 기본값 명시
{
    // 입력 포인터 유효성 검사 (옵션)
    if (!new_pc) {
        ROS_ERROR("publish_points called with null pointer!");
        return;
    }

    // is_dense 설정 (pc_out 생성 시 설정하는 것이 더 나을 수 있음)
    // new_pc->is_dense = true;

    // ROS 메시지로 변환
    sensor_msgs::PointCloud2 pc_new_msg;
    pcl::toROSMsg(*new_pc, pc_new_msg); // 포인터 역참조하여 전달

    // 헤더 정보 복사 및 프레임 ID 설정
    pc_new_msg.header = old_msg.header;
    pc_new_msg.header.frame_id = "velodyne"; // 원하는 출력 프레임 ID

    // 타임스탬프 업데이트 로직
    // update_timestamp가 0이 아닐 경우에만 헤더 타임스탬프 덮어쓰기
    if (update_timestamp != 0.0) { // 0.0과 비교
        pc_new_msg.header.stamp = ros::Time().fromSec(update_timestamp);
    }
    // else: old_msg.header.stamp (원본 헤더 타임스탬프) 사용

    // 메시지 발행
    pubRobosensePC.publish(pc_new_msg);
}

// --- RoboSense Handlers (기존 코드 - 일부 수정) ---

// RoboSense XYZI 타입 핸들러 (기존 코드)
void rsHandler_XYZI(const sensor_msgs::PointCloud2 &pc_msg) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr pc(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<VelodynePointXYZIR>::Ptr pc_new(new pcl::PointCloud<VelodynePointXYZIR>()); // XYZIR로 출력
    pcl::fromROSMsg(pc_msg, *pc);

    pc_new->points.reserve(pc->points.size());

    // to new pointcloud
    for (const auto& point : pc->points) { // range-based for loop 사용
        if (has_nan(point))
            continue;

        VelodynePointXYZIR new_point;
        new_point.x = point.x;
        new_point.y = point.y;
        new_point.z = point.z;
        new_point.intensity = point.intensity;
        // remap ring id
        // Note: pc->height and pc->width might not be reliable if input is not organized
        // Consider checking pc_msg.height and pc_msg.width instead, or alternative logic
        if (pc_msg.height == 16) { // Use pc_msg dimensions
            // Assuming point_id logic holds if organized
            // This logic needs verification for unorganized clouds or specific drivers
            size_t point_id = &point - &pc->points[0]; // Calculate index (if needed)
             if (pc_msg.width > 0) { // Avoid division by zero
                 new_point.ring = RING_ID_MAP_16[point_id / pc_msg.width];
             } else {
                 new_point.ring = 0; // Default or error handling
             }
        } else if (pc_msg.height == 128) {
             size_t point_id = &point - &pc->points[0];
             new_point.ring = RING_ID_MAP_RUBY[point_id % pc_msg.height];
        } else {
            // Handle other cases or default ring value if necessary
            new_point.ring = 0;
        }
        pc_new->points.push_back(new_point);
    }
    pc_new->width = pc_new->points.size();
    pc_new->height = 1;
    pc_new->is_dense = true; // Assuming NaNs are filtered

    publish_points(pc_new, pc_msg); // Pass Ptr directly, use original header time
}

// 데이터 복사 템플릿 함수 (기존 코드)
template<typename T_in_p, typename T_out_p>
void handle_pc_msg(const typename pcl::PointCloud<T_in_p>::Ptr &pc_in,
                   const typename pcl::PointCloud<T_out_p>::Ptr &pc_out) {
    pc_out->points.reserve(pc_in->points.size());
    // to new pointcloud
    for (const auto& point : pc_in->points) { // range-based for loop
        if (has_nan(point))
            continue;
        T_out_p new_point;
        new_point.x = point.x;
        new_point.y = point.y;
        new_point.z = point.z;
        new_point.intensity = point.intensity;
        // Note: Ring and Time fields are NOT copied here. They are handled by add_ring/add_time
        pc_out->points.push_back(new_point);
    }
    pc_out->width = pc_out->points.size();
    pc_out->height = 1;
    pc_out->is_dense = true;
}

// Ring 정보 추가 함수 (기존 코드)
template<typename T_in_p, typename T_out_p>
void add_ring(const typename pcl::PointCloud<T_in_p>::Ptr &pc_in,
              const typename pcl::PointCloud<T_out_p>::Ptr &pc_out) {
    // Assumes pc_out already has the correct points (from handle_pc_msg)
    // and we just need to add the ring field. Requires pc_out->points to be non-empty
    if (pc_out->points.empty()) return;

    int valid_point_id = 0;
    for (const auto& point_in : pc_in->points) {
        if (has_nan(point_in))
            continue;
        // Check bounds before accessing pc_out
        if (valid_point_id < pc_out->points.size()) {
             pc_out->points[valid_point_id++].ring = point_in.ring;
        } else {
             // Handle error: mismatch in number of valid points
             ROS_ERROR("Mismatch in valid point count during add_ring");
             break;
        }
    }
}

// Time 정보 추가 함수 (기존 코드)
template<typename T_in_p, typename T_out_p>
void add_time(const typename pcl::PointCloud<T_in_p>::Ptr &pc_in,
              const typename pcl::PointCloud<T_out_p>::Ptr &pc_out, double& msg_frame_time) {
    // Assumes pc_out already has the correct points (from handle_pc_msg)
    if (pc_out->points.empty()) return;

    double start_fire_time = std::numeric_limits<double>::max(); // Initialize properly
    for (const auto& point_in : pc_in->points) {
        if (!has_nan(point_in) && point_in.timestamp != 0 && point_in.timestamp < start_fire_time) {
            start_fire_time = point_in.timestamp;
        }
    }

    // Use scan start time if no valid points found or all timestamps are 0
    if (start_fire_time == std::numeric_limits<double>::max()) {
        start_fire_time = pc_in->points.empty() ? msg_frame_time : pc_in->points[0].timestamp; // Fallback
        // ROS_WARN("Could not determine valid start_fire_time, using fallback.");
    }

    int valid_point_id = 0;
    double first_point_timestamp = pc_in->points.empty() ? start_fire_time : pc_in->points[0].timestamp; // Use first point's timestamp as reference

    for (const auto& point_in : pc_in->points) {
        if (has_nan(point_in))
            continue;

        // Check bounds before accessing pc_out
        if (valid_point_id < pc_out->points.size()) {
            // Calculate relative time in seconds
            pc_out->points[valid_point_id++].time = static_cast<float>(point_in.timestamp - first_point_timestamp);
        } else {
             ROS_ERROR("Mismatch in valid point count during add_time");
             break;
        }

    }

    // Update frame time to the *earliest* firing time found
    msg_frame_time = start_fire_time;
}

// RoboSense XYZIRT 타입 핸들러 (기존 코드 - 약간 수정)
void rsHandler_XYZIRT(const sensor_msgs::PointCloud2 &pc_msg) {
    pcl::PointCloud<RsPointXYZIRT>::Ptr pc_in(new pcl::PointCloud<RsPointXYZIRT>());
    try {
        pcl::fromROSMsg(pc_msg, *pc_in);
    } catch (const std::exception& e) {
        ROS_ERROR("RoboSense XYZIRT: Failed to convert ROS message: %s", e.what());
        return;
    }


    if (output_type == "XYZIRT") {
        pcl::PointCloud<VelodynePointXYZIRT>::Ptr pc_out(new pcl::PointCloud<VelodynePointXYZIRT>());
        handle_pc_msg<RsPointXYZIRT, VelodynePointXYZIRT>(pc_in, pc_out); // Copy base fields
        add_ring<RsPointXYZIRT, VelodynePointXYZIRT>(pc_in, pc_out);      // Add ring
        double frame_start_T = pc_msg.header.stamp.toSec();
        add_time<RsPointXYZIRT, VelodynePointXYZIRT>(pc_in, pc_out, frame_start_T); // Add relative time & update frame_start_T
        // Publish based on use_frame_time flag
        if (use_frame_time){
            publish_points(pc_out, pc_msg); // Use original header time
        } else {
            publish_points(pc_out, pc_msg, frame_start_T); // Use calculated earliest fire time
        }
    } else if (output_type == "XYZIR") {
        pcl::PointCloud<VelodynePointXYZIR>::Ptr pc_out(new pcl::PointCloud<VelodynePointXYZIR>());
        handle_pc_msg<RsPointXYZIRT, VelodynePointXYZIR>(pc_in, pc_out);
        add_ring<RsPointXYZIRT, VelodynePointXYZIR>(pc_in, pc_out);
        publish_points(pc_out, pc_msg); // Use original header time
    } else if (output_type == "XYZI") {
        pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out(new pcl::PointCloud<pcl::PointXYZI>());
        handle_pc_msg<RsPointXYZIRT, pcl::PointXYZI>(pc_in, pc_out); // Only copies x,y,z,intensity
        publish_points(pc_out, pc_msg); // Use original header time
    } else {
        ROS_WARN("Unsupported output_type '%s' for RoboSense XYZIRT input.", output_type.c_str());
    }
}


// --- ML-X Handler (Workaround - 추가) ---
void mlxHandlerWorkaround(const sensor_msgs::PointCloud2 &pc_msg) {
    // 입력 클라우드: MLXPointXYZRGBF 사용 (rosbag의 x,y,z,rgb(float) 포맷)
    pcl::PointCloud<MLXPointXYZRGBF>::Ptr pc_in(new pcl::PointCloud<MLXPointXYZRGBF>());
    try {
        pcl::fromROSMsg(pc_msg, *pc_in);
    } catch (const std::exception& e) {
        ROS_ERROR("MLX Workaround: Failed to convert ROS message to MLXPointXYZRGBF: %s", e.what());
        // 추가 디버깅 정보 출력 (옵션)
        ROS_ERROR("Input message fields:");
        for(const auto& field : pc_msg.fields) {
             ROS_ERROR("  name: %s, offset: %u, datatype: %u, count: %u", field.name.c_str(), field.offset, field.datatype, field.count);
        }
        ROS_ERROR("Input message point_step: %u", pc_msg.point_step);
        return; // 변환 실패 시 처리 중단
    }

    // 출력 클라우드: VelodynePointXYZIRT 사용 (PV-LIO가 읽도록)
    pcl::PointCloud<VelodynePointXYZIRT>::Ptr pc_out(new pcl::PointCloud<VelodynePointXYZIRT>());
    pc_out->points.reserve(pc_in->points.size()); // 메모리 미리 할당

    // 포인트 변환 및 필드 설정 (Workaround 적용)
    for (const auto& pt_in : pc_in->points) {
        if (has_nan(pt_in)) { // Check only x, y, z which are present
            continue;
        }

        VelodynePointXYZIRT pt_out;
        pt_out.x = pt_in.x;
        pt_out.y = pt_in.y;
        pt_out.z = pt_in.z;

        // --- Workaround 적용 ---
        pt_out.intensity = 0.0f; // Intensity 0으로 설정
        pt_out.ring = 0;         // Ring 0으로 설정
        pt_out.time = 0.0f;      // 상대 시간을 0으로 설정 (Base Timestamp 사용 효과)
        // -----------------------

        pc_out->points.push_back(pt_out);
    }

    // 포인트 클라우드 메타데이터 설정
    if (!pc_out->points.empty()) {
        pc_out->width = pc_out->points.size();
        pc_out->height = 1;
        pc_out->is_dense = true; // NaN 제거 가정
    } else {
        // Handle empty cloud case if necessary
        pc_out->width = 0;
        pc_out->height = 1;
        pc_out->is_dense = true;
    }

    // 결과 퍼블리시 (Workaround 버전 - 원본 헤더 타임스탬프 사용)
    sensor_msgs::PointCloud2 pc_new_msg;
    pcl::toROSMsg(*pc_out, pc_new_msg);
    pc_new_msg.header = pc_msg.header; // 원본 헤더 사용
    pc_new_msg.header.frame_id = "velodyne"; // PV-LIO가 기대하는 frame_id
    pubRobosensePC.publish(pc_new_msg);
}


// --- Main Function (수정) ---
int main(int argc, char **argv) {
    ros::init(argc, argv, "rs_converter"); // 노드 이름 변경 (더 일반적)
    ros::NodeHandle nh;

    if (argc < 3) {
        ROS_ERROR("Usage: rosrun <pkg> rs_converter <input_type> <output_type> [use_frame_time]");
        ROS_ERROR("  <input_type>: XYZI | XYZIRT (for RoboSense) | MLX (for ML-X Workaround)");
        ROS_ERROR("  <output_type>: XYZI | XYZIR | XYZIRT (ignored for MLX input)");
        ROS_ERROR("  [use_frame_time]: true | false (optional, defaults to false, ignored for MLX input)");
        exit(1);
    }

    std::string input_type = argv[1];
    output_type = argv[2]; // RoboSense 경우에 사용, MLX 경우 무시됨

    // use_frame_time 파싱 (MLX 경우는 무시됨)
    use_frame_time = false; // 기본값
    if (argc == 4 && std::strcmp("true", argv[3]) == 0) {
        use_frame_time = true;
    }
    if (input_type != "MLX") { // RoboSense 경우에만 로그 출력
         ROS_INFO("Use Frame Time is %s.", use_frame_time ? "ON" : "OFF");
    }

    // 출력 토픽 설정
    pubRobosensePC = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_points", 1);

    // 입력 타입에 따라 구독 설정
    if (input_type == "XYZI") {
        subRobosensePC = nh.subscribe("/rslidar_points", 1, rsHandler_XYZI);
        ROS_INFO("Input: RoboSense XYZI (/rslidar_points), Output: %s (/velodyne_points)", output_type.c_str());
    } else if (input_type == "XYZIRT") {
        subRobosensePC = nh.subscribe("/rslidar_points", 1, rsHandler_XYZIRT);
        ROS_INFO("Input: RoboSense XYZIRT (/rslidar_points), Output: %s (/velodyne_points)", output_type.c_str());
    } else if (input_type == "MLX") {
        // MLX Workaround 사용 시, 입력 토픽 이름 주의 (/ml_/pointcloud 사용 가정)
        subRobosensePC = nh.subscribe("/ml_/pointcloud", 1, mlxHandlerWorkaround);
        ROS_WARN("Input: ML-X WORKAROUND (/ml_/pointcloud), Output: VelodynePointXYZIRT (time=0) (/velodyne_points)");
        ROS_WARN("!!! WARNING: Using MLX workaround. Motion undistortion in PV-LIO will be disabled due to lack of timestamps in the input bag file. LIO performance may be degraded. !!!");
    } else {
        ROS_ERROR("Unsupported input_type: %s. Use XYZI, XYZIRT, or MLX.", input_type.c_str());
        exit(1);
    }

    ros::spin();
    return 0;
}