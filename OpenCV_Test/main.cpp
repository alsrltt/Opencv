/*#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <time.h>
using namespace std;
using namespace cv;

uchar bilinear_value(Mat img, double x, double y) {
	if (x >= img.cols - 1)x--;
	if (y >= img.rows - 1)y--;

	Point pt((int)x, (int)y);
	int A = img.at<uchar>(pt);            // 좌측상단점
	int B = img.at<uchar>(pt + Point(0, 1)); //좌측하단점
	int C = img.at<uchar>(pt + Point(1, 0));  //우측상단점
	int D = img.at<uchar>(pt + Point(1, 1));  //우측하단점

	double alpha = y - pt.y;
	double beta = x - pt.x;
	int M1 = A + (int)cvRound(alpha*(B - A));
	int M2 = C + (int)cvRound(alpha*(D - C));
	int P = M1 + (int)cvRound(beta*(M2 - M1));
	return saturate_cast<uchar>(P);
}
void rotation(Mat img, Mat& dst, double dgree) {
	double radian = dgree / 180 * CV_PI;
	double sin_value = sin(radian);
	double cos_value = cos(radian);

	Rect rect(Point(0, 0), img.size());
	dst = Mat(img.size(), img.type(), Scalar(0));
	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			double x = j * cos_value + i * sin_value;
			double y = -j * sin_value + i * cos_value;
			if (rect.contains(Point2d(x, y))) {
				dst.at<uchar>(i, j) = bilinear_value(img, x, y);
			}
		}
	}
}
void rotation(Mat img, Mat& dst, double dgree, Point pt) {
	double radian = dgree / 180 * CV_PI;
	double sin_value = sin(radian);
	double cos_value = cos(radian);
	Rect rect(Point(0, 0), img.size());
	dst = Mat(img.size(), img.type(), Scalar(0));
	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			int jj = j - pt.x;
			int ii = i - pt.y;
			double x = jj * cos_value + ii * sin_value + pt.x;
			double y = -jj * sin_value + ii * cos_value + pt.y;
			if (rect.contains(Point2d(x, y))) {
				dst.at<uchar>(i, j) = bilinear_value(img, x, y);
			}
		}
	}

}

int main() {
	Mat image = imread("./image/pep.jpg", 0);
	CV_Assert(image.data);
	Mat dst1, dst2;
	Point center = image.size() / 2;
	rotation(image, dst1, 20);
	rotation(image, dst2, 20, Point(image.size().width/2, image.size().height/2));
	imshow("image", image);
	imshow("원점중심 회전20도", dst1);
	imshow("이미지 중심 회전 20도", dst2);
	waitKey(0);
	return 0;
}*/

#include "preprocess.hpp"

Point2d calc_center(Rect obj) {
	Point2d c = (Point2d)obj.size() / 2.0;
	Point2d center = (Point2d)obj.tl() + c;
	return center;
}
Rect face_tmp;
int main() {
	CascadeClassifier face_cascade, eyes_cascade;
	load_cascade(face_cascade, "haarcascade_frontalface_alt2.xml");
	load_cascade(eyes_cascade, "haarcascade_eye.xml");



	//Mat image = imread("./image/joa.jpg", IMREAD_COLOR);
	//CV_Assert(image.data);
	//resize(image, image, Size(0, 0), 4, 4, INTER_LINEAR);
	VideoCapture capture;
	capture.open(0);
	Mat frame;
	Mat image;
	int delay = 30;
	capture.set(CAP_PROP_FRAME_WIDTH, 50);
	capture.set(CAP_PROP_FRAME_HEIGHT, 50);

	while (1) {
		capture.read(image);
		CV_Assert(image.data);
		flip(image, image, 1);
		Mat gray = preprocessing(image);
		vector<Rect> faces, eyes;
		vector<Point2d> eyes_center;
		
		face_cascade.detectMultiScale(gray, faces, 1.1, 1, 0, Size(5, 5));
		if (faces.size() > 0) {
			eyes_cascade.detectMultiScale(gray(faces[0]), eyes, 1.15, 7, 0, Size(25, 20));
			cout << faces.size() << endl;
			if (eyes.size() == 2) {
				for (size_t i = 0; i < eyes.size(); i++) {
					Point2d center = calc_center(eyes[i] + faces[0].tl());
					circle(image, center, 5, Scalar(0, 255, 0), 2);
				}
			}
			for (int i = 0;i < faces.size();i++)
				rectangle(image, faces[i], Scalar(255, 0, 0), 2);
			face_tmp = faces[0];
			imshow("image", image);
			if (waitKey(delay) >= 0)break;
		}
		else {
			
			rectangle(image, face_tmp, Scalar(0, 0, 255), 2);
			imshow("image", image);if (waitKey(delay) >= 0)break;
		}
	}
	return 0;
}