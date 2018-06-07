
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <time.h>
using namespace std;
using namespace cv;

void draw_houghLines(Mat src, Mat &dst, vector<Vec2f> lines, int nline) {
	cvtColor(src, dst, CV_GRAY2BGR);
	for (size_t i = 0; i < min((int)lines.size(), nline); i++) {
		float rho = lines[i][0], theta = lines[i][1];
		double a = cos(theta), b = sin(theta);
		Point2d pt(a*rho, b*rho);
		Point2d delta(1000 * -b, 1000 * a);
		line(dst, pt + delta, pt - delta, Scalar(0, 255, 0), 1, LINE_AA);
	}
}

int main() {
	VideoCapture capture("./image/vedio1.avi");
	if (!capture.isOpened()) {
		exit(1);
	}

	double rho = 1, theta = CV_PI / 180;
	Mat frame;
	Mat canny, edge, dst1;
	double fps = 29.97;
	Size size;
	int fourcc = VideoWriter::fourcc('D', 'X', '5', '0');
	capture.read(frame);
	VideoWriter writer;
	writer.open("./image/vedio.avi", fourcc, fps, frame.size());
	CV_Assert(writer.isOpened());
	while (capture.read(frame)) {
		CV_Assert(frame.data);
		GaussianBlur(frame, canny, Size(5, 5), 1, 1);
		Canny(canny, edge, 100, 150);
		vector<Vec2f> lines;
		HoughLines(edge, lines, rho, theta, 130);
		draw_houghLines(edge, dst1, lines, 10);

		imshow("frame", frame);
		imshow("hough", dst1);
		writer << dst1;
		if (waitKey(30) >= 0) break;
	}
	return 0;
}
