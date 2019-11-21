#include <iostream>
#include "pcl/io/pcd_io.h"
#include "pcl/visualization/pcl_visualizer.h"
#include "Eigen/Core"
#include "Eigen/Dense"
#include "pcl/common/centroid.h"
#include "pcl/common/transforms.h"
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>

pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr selected_point_(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointXYZ current_point_(0.0,0.0,0.0);
pcl::visualization::PCLVisualizer viewer("Viewer");

Eigen::Quaterniond TwoAxistoQuaterniond(Eigen::Vector3d v1, Eigen::Vector3d v2){
    v1 = v1/v1.norm();
    v2 = v2/v2.norm();
    if((v1-v2).norm() < 0.00001 || (v1+v2).norm() < 0.00001) {
        return Eigen::Quaterniond::Identity();
    }
    Eigen::Vector3d u = v1.cross(v2);
    u = u/u.norm();
    double theta = acos(v1.cwiseProduct(v2).sum())/2;
    return Eigen::Quaterniond(cos(theta), u[0]*sin(theta), u[1]*sin(theta),u[2]*sin(theta)).normalized();
}
Eigen::Matrix3d CalURDF(){
    if(selected_point_->size()<3){
        std::cout<<"selected points num too small."<<std::endl;
        return Eigen::Matrix3d::Identity();
    }
    Eigen::Matrix3d covariance_matrix;
    Eigen::Vector4d centroid;
//    pcl::compute3DCentroid(*selected_point_, centroid);
//    pcl::computeCovarianceMatrix(*selected_point_, centroid, covariance_matrix);
    pcl::computeMeanAndCovarianceMatrix(*selected_point_, covariance_matrix, centroid);
    Eigen::Vector3d::Scalar eigen_value;
    Eigen::Vector3d eigen_vector;
    pcl::eigen33 (covariance_matrix, eigen_value, eigen_vector);
    if(eigen_vector[1]<0){
        eigen_vector = -eigen_vector;
    }
    Eigen::Vector3d zaxis(0.0, 1.0, 0.0);
    Eigen::Quaterniond qua = TwoAxistoQuaterniond(eigen_vector, zaxis);
    Eigen::Matrix3d rotation_matrix = qua.toRotationMatrix();
    Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(M_PI_2,Eigen::Vector3d::UnitX()));
    Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(M_PI,Eigen::Vector3d::UnitZ()));
    Eigen::Matrix3d rotation = yawAngle*rollAngle*rotation_matrix;

    Eigen::Vector3d eulerAngle = rotation.eulerAngles(0,1,2);
    std::cout<<"URDF: "<<eulerAngle<<std::endl;
    return rotation;
}
void PointPickEventOccured(const pcl::visualization::PointPickingEvent &event){
    float x, y, z;
    event.getPoint(x, y, z);
    current_point_.x = x;
    current_point_.y = y;
    current_point_.z = z;
    viewer.removePointCloud("current cloud");
    pcl::PointCloud<pcl::PointXYZ> current;
    current.push_back(current_point_);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> current_h(current.makeShared(), 0, 255, 0);
    viewer.addPointCloud(current.makeShared(), current_h, "current cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "current cloud");
}

void KeyboardEventOccured(const pcl::visualization::KeyboardEvent &event){
    if (event.getKeySym() == "s" && event.keyDown()){
        selected_point_->points.push_back(current_point_);
        viewer.removePointCloud("select cloud");
        viewer.removePointCloud("current cloud");
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> select_h(selected_point_, 255, 0, 0);
        viewer.addPointCloud(selected_point_, select_h, "select cloud");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "select cloud");
    }
    if (event.getKeySym() == "x" && event.keyDown()) {
        if(!selected_point_->empty()) {
            selected_point_->points.pop_back();
            viewer.removePointCloud("select cloud");
            viewer.removePointCloud("current cloud");
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> select_h(selected_point_, 0, 0, 255);
            viewer.addPointCloud(selected_point_, select_h, "select cloud");
            viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "select cloud");
        }else{
            std::cout<<"Can't delete."<<std::endl;
        }
    }
    if(event.getKeySym() == "space" && event.keyDown()){
        Eigen::Matrix3d rotation_matrix = CalURDF();
        Eigen::Matrix4d trans = Eigen::Matrix4d::Identity();
        trans.block<3,3>(0, 0) = rotation_matrix;
        trans.block<3,1>(0, 3) = Eigen::Vector3d(0.0, 0.0, 0.0);
        trans.block<1,3>(3, 0) = Eigen::Vector3d(0.0, 0.0, 0.0);
        trans(3,3) = 1.0;
        viewer.removePointCloud("select cloud");
        viewer.removePointCloud("current cloud");
        pcl::PointCloud<pcl::PointXYZ>::Ptr trans_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*cloud_, *trans_cloud, trans);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> trans_h(trans_cloud, 0, 255, 0);
        viewer.addPointCloud(trans_cloud, trans_h, "trans cloud");
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "trans cloud");
        viewer.addCoordinateSystem();
    }
}

void PlaneModelSegmentationURDF(){
    viewer.removeAllPointClouds();
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (0.01);
    seg.setInputCloud (cloud_);
    seg.segment (*inliers, *coefficients);
    for (std::size_t i = 0; i < inliers->indices.size (); ++i) {
        selected_point_->push_back(cloud_->points[inliers->indices[i]]);
    }
    Eigen::Matrix3d rotation_matrix = CalURDF();
    Eigen::Matrix4d trans = Eigen::Matrix4d::Identity();
    trans.block<3,3>(0, 0) = rotation_matrix;
    trans.block<3,1>(0, 3) = Eigen::Vector3d(0.0, 0.0, 0.0);
    trans.block<1,3>(3, 0) = Eigen::Vector3d(0.0, 0.0, 0.0);
    trans(3,3) = 1.0;

    pcl::PointCloud<pcl::PointXYZ>::Ptr trans_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*cloud_, *trans_cloud, trans);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> trans_h(trans_cloud, 0, 255, 0);
    viewer.addPointCloud(trans_cloud, trans_h, "trans cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "trans cloud");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> plane_h(selected_point_, 255, 0, 0);
    viewer.addPointCloud(selected_point_, plane_h, "plane cloud");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "plane cloud");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_h(cloud_, 255, 255, 255);
    viewer.addPointCloud(cloud_, cloud_h, "original cloud");

    viewer.addCoordinateSystem();
}
void ManualPickPointURDF(){
    selected_point_->clear();
    viewer.removeAllPointClouds();
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_h(cloud_, 255, 255, 255);
    viewer.addPointCloud(cloud_, cloud_h, "original cloud");
    viewer.registerPointPickingCallback(&PointPickEventOccured);
    viewer.registerKeyboardCallback(&KeyboardEventOccured);
}
int main() {
    std::string file_path = "..//data//data.pcd";
    pcl::io::loadPCDFile<pcl::PointXYZ>(file_path, *cloud_);
    std::string key;
    std::cout<<"使用平面模型?(y/n):"<<std::endl;
    getline(cin, key, '\n');
    bool useModel = true;
    if(key == "n" || key == "N")
        useModel = false;
    if(useModel)
        PlaneModelSegmentationURDF();
    else {
        std::cerr << "Select Current Point:  Shift + Left-click" << std::endl;
        std::cerr << "Confirm Points:        s" << std::endl;
        std::cerr << "Delete Last Points:    x" << std::endl;
        std::cerr << "Calculate URDF:        space" << std::endl;
        ManualPickPointURDF();
    }

    while (!viewer.wasStopped())
    {
        viewer.spinOnce(10);
    }
    return 0;
}