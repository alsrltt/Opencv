#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h> 
#include <time.h>

using namespace cv;
using namespace std;
string title = "트랙바 이벤트";
Mat image;
static Point pt(-1, -1);

void onMouse(int event, int x, int y, int flags, void *p) {
	
	if (event == EVENT_LBUTTONDOWN) {
		if (pt.x < 0) {
			pt = Point(x, y);
		}
		else {
			rectangle(image, pt, Point(x, y), Scalar(255, 100, 25), 3);
			pt = Point(-1, -1);
		}
	}
}

int main(int argc, char** argv)
{	
	Mat image(300, 400, CV_8UC3, Scalar(55, 15, 10));
	imshow(title, image);
	
	setMouseCallback(title, onMouse, 0);
	
	
	waitKey();
}
