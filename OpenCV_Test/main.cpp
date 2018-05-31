/*#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
string title = "blur";
Mat image;
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
*//*
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
*//*
void onMouse(int event, int x, int y, int flags, void *param) {
	static Point pt(-1, -1);
	if (event == EVENT_LBUTTONDOWN) {
		if (pt.x < 0)pt = Point(x, y);
		else {
			Mat blur;
			cout << "blurring" << endl;
			float data[] = {
				1 / 9.f, 1 / 9.f, 1 / 9.f,
				1 / 9.f, 1 / 9.f, 1 / 9.f
				,1 / 9.f, 1 / 9.f, 1 / 9.f
			};
			Mat mask(3, 3, CV_32F, data);
			Rect rect(pt, Point(x, y));
			Mat roi = image(rect);
			filter1(roi, blur, mask);
			blur.convertTo(blur, CV_8U);
			blur.copyTo(roi);

			imshow(title, image);
			pt = Point(-1, -1);
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
	/*
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
	*//*
	return 0;
}*/
/*
#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>

using namespace std;
using namespace cv;

int boundary = 25;
Mat image, edge;

void differOp(Mat img, Mat& dst, int mask_size) {
	dst = Mat(img.size(), CV_8U, Scalar(0));
	Point h_m(mask_size / 2, mask_size / 2);
	int k = 0;
	int mask_length = mask_size * mask_size;
	for (int i = h_m.y; i < img.rows - h_m.y; i++) {
		for (int j = h_m.x; j < img.cols - h_m.x; j++) {
			vector <uchar> mask(mask_length, 0);
			float Max = 0;
			for (int u = 0;u < mask_size;u++) {
				for (int v = 0;v < mask_size;v++, k++) {
					int y = i + u - h_m.y;
					int x = i + v - h_m.x;
					mask[k] = img.at<uchar>(y, x);
				}
			}
			uchar max = 0;
			for (int k = 0;k < mask_length / 2;k++) {
				int start = mask[k];
				int end = mask[mask_length - 1 - k];
				uchar difference = abs(end - start);
				if (difference > max)max = difference;
			}
			dst.at<uchar>(i, j) = max;
		}
	}
}

void homogenOp(Mat img, Mat& dst, int mask_size) {
	int value = 0;
	dst = Mat(img.size(), CV_8U, Scalar(0));
	Point h_m(mask_size / 2, mask_size / 2);
	int mask_length = mask_size * mask_size;
	for (int i = h_m.y; i < img.rows - h_m.y; i++) {
		for (int j = h_m.x; j < img.cols - h_m.x; j++) {
			vector<uchar> mask(mask_length, 0);
			float Max = 0;
			for (int u = 0; u < mask_size; u++) {
				for (int v = 0; v < mask_size; v++) {
					int y = i + u - h_m.y;
					int x = j + v - h_m.x;
					float difference = abs(img.at<uchar>(i, j) - img.at<uchar>(y, x));
					Max = max(difference, Max);
				}
			}
			if (Max >= boundary)
				value = 255;
			else
				value = 0;
			dst.at<uchar>(i, j) = value;
		}
	}
}
void onChanged(int value, void * userData) {
	homogenOp(image, edge, 3);
	imshow("edge", edge);
}

int main() {
	image = imread("./image/image.jpg", IMREAD_GRAYSCALE);
	CV_Assert(image.data);
	differOp(image, edge, 3);
	imshow("image", image);
	imshow("edge", edge);
	waitKey(0);
}*/

/*
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <time.h>
using namespace std;
using namespace cv;
Mat image;
Mat edge;

void homogenOp(Mat img, Mat& dst, int mask_size) {
	dst = Mat(img.size(), CV_32F, Scalar(0));
	Point h_m(mask_size / 2, mask_size/2);

	for (int i = h_m.y; i < img.rows - h_m.y; i++) {
		for (int j = h_m.x; j < img.cols - h_m.x; j++) {

			float max = 0;
			for (int u = 0; u < mask_size; u++) {
				for (int v = 0; v < mask_size; v++) {
					int y = i + u - h_m.y;
					int x = j + v - h_m.x;
					float difference = abs(img.at<uchar>(i, j) - img.at<uchar>(y, x));
					if (difference > max)max = difference;
				}
			}
			dst.at<uchar>(i, j) = max;
		}
	}
}

int main() {
	image = imread("./image/money.jpg", IMREAD_GRAYSCALE);
	CV_Assert(image.data);
	homogenOp(image, edge, 3);
	imshow("image", image);
	imshow("edge", edge);
	waitKey(0);
	return 0;
}
*/
/*
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <math.h>
#include <time.h>
#include <Windows.h>
#include <cstring>

using namespace cv;
using namespace std;

Mat hsv;
Mat bin;
int Hue;
Mat image;

void onMouse(int event, int x, int y, int flags, void *param) {
	switch (event) {
	case EVENT_LBUTTONDOWN:
		int r, g, b;
		b = image.at<Vec3b>(y, x)[0];
		g = image.at<Vec3b>(y, x)[1];
		r = image.at<Vec3b>(y, x)[2];

		Mat color = Mat(200, 200, CV_8UC3);
		color = Scalar(b, g, r);

		imshow("color", color);
		//cout << x << endl;
		//cout << y << endl;
		cout << int(hsv.at<Vec3b>(y, x)[0]) << endl;
		Hue = int(hsv.at<Vec3b>(y, x)[0]);

	}
}

int main() {
	image = imread("./image/pep.jpg", IMREAD_COLOR);
	Mat show = Mat(image.cols, image.rows, CV_8UC1);
	cvtColor(image, hsv, CV_RGB2HSV);

	while (true) {

		imshow("Image", image);
		setMouseCallback("Image", onMouse, NULL);

		int Low = Hue - 5;
		int High = Hue + 5;

		for (int i = 0; i < image.cols; i++) {
			for (int j = 0; j < image.rows; j++) {
				if (Low < int(hsv.at<Vec3b>(i, j)[0]) && High > int(hsv.at<Vec3b>(i, j)[0])) {
					show.at<uchar>(i, j) = 255;
				}
				else {
					show.at<uchar>(i, j) = 0;
				}
			}
		}
		imshow("show", show);
		waitKey(50);
	}
	waitKey(0);
	return 0;
}
*/
/*
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <time.h>
using namespace std;
using namespace cv;

string title = "blure";

Mat image;


void filter1(Mat img, Mat& dst, Mat mask) {
	img.copyTo(dst);
	//	dst = Mat(img.size(), CV_32F, Scalar(0));
	Point h_m = mask.size() / 2;

	for (int i = h_m.y; i < img.rows - h_m.y; i++) {
		for (int j = h_m.x; j < img.cols - h_m.x; j++) {

			float sum = 0;
			for (int u = 0; u < mask.rows; u++) {
				for (int v = 0; v < mask.cols; v++) {
					int y = i + u - h_m.y;
					int x = j + v - h_m.x;
					sum += mask.at<float>(u, v)*img.at<uchar>(y, x);
				}
			}
			dst.at<uchar>(i, j) = sum;
		}
	}
}

void differential(Mat image, Mat& dst, float data1[], float data2[]) {
	Mat dst1, dst2;
	Mat mask1(3, 3, CV_32F, data1); //135도 대각선
	Mat mask2(3, 3, CV_32F, data2); //45도 대각선'
	filter1(image, dst1, mask1);
	filter1(image, dst2, mask2);
	magnitude(dst1, dst2, dst);
	dst1 = abs(dst1);
	dst2 = abs(dst2);
	dst.convertTo(dst, CV_8U);//대각선 양측
	dst1.convertTo(dst1, CV_8U);
	dst2.convertTo(dst2, CV_8U);
	imshow("dst1", dst1);
	imshow("dst2", dst2);
}

void differential2(Mat image, Mat& dst, float data1[], float data2[]) {
	Mat dst1, dst2;
	Mat mask1(3, 3, CV_32F, data1); //135도 대각선
	Mat mask2(3, 3, CV_32F, data2); //45도 대각선'
	filter2D(image, dst1, CV_32F, mask1); // openCV 내장 함수
	filter2D(image, dst2, CV_32F, mask2);
	magnitude(dst1, dst2, dst);
	dst.convertTo(dst, CV_8U);//수직 수평
	convertScaleAbs(dst1, dst1);//수직
	convertScaleAbs(dst2, dst2);//수평
	imshow("dst1", dst1);
	imshow("dst2", dst2);
}


void onMouse(int event, int x, int y, int flags, void*param) {
	static Point pt(-1, -1);
	if (event == EVENT_LBUTTONDOWN) {

		if (pt.x < 0)
			pt = Point(x, y);
		else {
			Mat blur;
			cout << "blurring" << endl;
			float data[] = {
				1 / 9.f,1 / 9.f,1 / 9.f,
				1 / 9.f,1 / 9.f,1 / 9.f,
				1 / 9.f,1 / 9.f,1 / 9.f
			};
			Mat mask(3, 3, CV_32F, data);
			Rect rect(pt, Point(x, y));
			Mat roi = image(rect);
			filter1(roi, blur, mask);
			blur.convertTo(blur, CV_8U);
			blur.copyTo(roi);
			imshow(title, image);
			pt = Point(-1, -1);
		}
	}
}
int main() {
	image = imread("./image/house (2).jpg", IMREAD_GRAYSCALE);
	CV_Assert(image.data);
	float data1[] = {
		-1, 0, 1,
		-2, 0, 2,
		-1, 0, 1
	};
	float data2[] = {
		-1, -2, -1, 
		0, 0, 0, 
		1, 2, 1
	};
	Mat dst;
	differential2(image, dst, data1, data2);
	imshow("image", image);
	imshow("dst", dst);
	waitKey(0);
	return 0;
}
*/
/*
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <time.h>
using namespace std;
using namespace cv;

void averageFilter(Mat img, Mat& dst, int size) {
	dst = Mat(img.size(), CV_8U, Scalar(0));
	for (int i = 0;i < img.rows;i++) {
		for (int j = 0;j < img.cols;j++) {
			Point pt1 = Point(i - size / 2, i - size / 2);
			Point pt2 = pt1 + (Point)Size(size, size);
			if (pt1.x < 0)pt1.x = 0;
			if (pt1.y < 0)pt1.y = 0;
			if (pt2.x > img.cols)pt2.x = img.cols;
			if (pt2.y > img.rows)pt2.y = img.rows;
			Rect mask_rect(pt1, pt2);
			Mat mask = img(mask_rect);
			dst.at<uchar>(i, j) = (uchar)mean(mask)[0];
		}
	}
}

void medianFilter(Mat img, Mat& dst, int size) {
	dst = Mat(img.size(), CV_8U, Scalar(0));
	Size msize(size, size);
	Point h_m = msize / 2;
	for(int i = h_m.y;i < img.rows - h_m.y;i++) {
		for (int j = h_m.x;j < img.cols - h_m.x;j++) {
			Point start = Point(j, i) - h_m;
			Rect roi(start, msize);
			Mat mask, sort_m;
			img(roi).copyTo(mask);
			Mat one_row = mask.reshape(1, 1);
			cv::sort(one_row, sort_m, SORT_EVERY_ROW);
			//sort_m << 정렬된 값
			int medi_idx = (int)(one_row.total() / 2);
				dst.at<uchar>(i, j) = sort_m.at<uchar>(medi_idx);
		}
	}
}

void minMaxFilter(Mat img, Mat& dst, int size, int flag = 1) {
	dst = Mat(img.size(), CV_8U, Scalar(0));
	Size msize(size, size);
	Point h_m = msize / 2;
	for (int i = h_m.y;i < img.rows - h_m.y;i++) {
		for (int j = h_m.x;j < img.cols - h_m.x;j++) {
			Point start = Point(j, i) - h_m;
			Rect roi(start, msize);
			Mat mask = img(roi);
			double minVal, maxVal;
			minMaxLoc(mask, &minVal, &maxVal);
			dst.at<uchar>(i, j) = (flag) ? maxVal : minVal;
		}
	}
}

int main() {
	Mat image = imread("./image/salt.jpg", 0);
	CV_Assert(image.data);
	Mat avg, med;
	averageFilter(image, avg, 3);
	medianFilter(image, med, 3);
	imshow("원본", image);
	imshow("최소", avg);
	imshow("최대", med);
	waitKey(0);
}
*/
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <time.h>
using namespace std;
using namespace cv;
int th = 50;
Mat image, gray, edge;
void onTrackbar(int value, void* userdata) {
	GaussianBlur(gray, edge, Size(3, 3), 0.7);
	Canny(edge, edge, th, th * 2);
	Mat color_edge;
	image.copyTo(color_edge, edge);
	imshow("edge", color_edge);
}

int main() {
	image = imread("./image/pep.jpg", 1);
	CV_Assert(image.data);
	cvtColor(image, gray, COLOR_BGR2GRAY);
	namedWindow("edge", 1);
	createTrackbar("canny th", "edge", &th, 255, onTrackbar);
	waitKey(0);
}