#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
/*
Range th(50, 100);
Mat hue, HSV;

void onThreshold(int value, void* userdata) {
	Mat result = Mat(hue.size(), CV_8U, Scalar(0));
	for (int i = 0;i < result.rows;i++) {
		for (int j = 0;j < result.cols;j++) {
			bool ck = hue.at<uchar>(i, j) >= th.start&&hue.at<uchar>(i, j) = (ck) ? 255 : 0;
		}
	}
	imshow("result", result); 
}

void onMouse(int event, int x, int y, int flags, void *param) {
	switch (event) {
	case EVENT_LBUTTONDOWN:
		cout << HSV.at<Vec3d>(y, x) << endl;
	}
}
*/

void filter1(Mat img, Mat& dst, Mat mask) {
	dst = Mat(img.size(), CV_32F, Scalar(0));
	Point h_m = mask.size() / 2;
	for (int i = h_m.y;i < img.rows - h_m.x;i++) {
		for (int j = h_m.x;j < img.cols - h_m.x;j++) {
			float sum = 0;
			for (int u = 0;u < mask.rows;u++) {
				for (int v = 0;v < mask.cols;v++) {
					int y = i + u - h_m.y;
					int x = j + v - h_m.x;
					sum += mask.at<float>(u, v)*img.at<uchar>(y, x);
				}
			}
			dst.at<float>(i, j) = sum;
		}
	}
}

int main() {
	/*Mat bgr_img = imread("./image/money.jpg", 1);
	CV_Assert(bgr_img.data);
	Mat hsv[3];
	cvtColor(bgr_img, HSV, CV_BGR2HSV);
	split(HSV, hsv); 
	hsv[0].copyTo(hue);
	namedWindow("result", WINDOW_AUTOSIZE);
	createTrackbar("hue_th1", "result", &th.start, 255, onThreshold);
	createTrackbar("hue_th2", "result", &th.end, 255, onThreshold);
	waitKey();*/

	Mat image = imread("./image/money.jpg", IMREAD_GRAYSCALE);
	CV_Assert(image.data);
	float data[] = {
		1 / 9.f, 1 / 9.f, 1 / 9.f,
		1 / 9.f, 1 / 9.f, 1 / 9.f
		, 1 / 9.f, 1 / 9.f, 1 / 9.f
	};
	float data2[] = {
		1/25.f, 1 / 25.f, 1 / 25.f , 1 / 25.f, 1 / 25.f, 
		1 / 25.f, 1 / 25.f, 1 / 25.f, 1 / 25.f, 1 / 25.f, 
		1 / 25.f, 1 / 25.f, 1 / 25.f, 1 / 25.f, 1 / 25.f,
		1 / 25.f, 1 / 25.f, 1 / 25.f, 1 / 25.f, 1 / 25.f,
		1 / 25.f, 1 / 25.f, 1 / 25.f, 1 / 25.f, 1 / 25.f
	};
	Mat mask(3, 3, CV_32F, data);
	Mat mask2(5, 5, CV_32F, data);

	Mat blur;
	Mat blur2;
	filter1(image, blur, mask);
	filter1(image, blur2, mask);
	blur.convertTo(blur, CV_8U);
	blur.convertTo(blur2, CV_8U);
	imshow("image", image);
	imshow("blur", blur);
	imshow("blur2", blur2);
	waitKey(0);
	return 0;
}