
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <time.h>
using namespace std;
using namespace cv;
Mat image = imread("./image/pep.jpg", 0);
Mat dst1;
void translation(Mat img, Mat& dst, Point pt) {
	Rect rect(Point(0, 0), img.size());
	dst = Mat(img.size(), img.type(), Scalar(0));
	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			Point dst_pt(j, i);
			Point img_pt = dst_pt - pt;
			if (rect.contains(img_pt))
				dst.at<uchar>(dst_pt) = img.at<uchar>(img_pt);
		}
	}
}
static Point pt(-1, -1);
static Point pt1(-1, -1);
static Point pt2(0, 0);
void onMouse(int event, int x, int y, int flags, void *userdata) {
	switch (event) {
	case EVENT_LBUTTONDOWN:
		if (pt.x < 0) {
			pt = Point(x, y);
		}
		else {
			pt1 = Point(x, y);
			pt2 = Point(pt.x - pt1.x, pt.y - pt1.y);
			translation(image, dst1, pt2);
			imshow("dst1", dst1);
			pt = Point(-1, -1);
		}

	}
}
int main() {
	CV_Assert(image.data);

	imshow("image", image);
	setMouseCallback("image", onMouse, 0);
	waitKey();
	return 0;
}