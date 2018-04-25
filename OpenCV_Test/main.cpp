
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h> 

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	Mat m1(2, 3, CV_8U<2);
	Mat m2(2, 3, CV_8U, Scalar(10));
	Mat m3 = m1 + m2;
	Mat m4 = m2 - 6;
	Mat m5 = m1;
	cout << "m1 = " << m1 << endl;
	cout << "m2 = " << m2 << endl;
	cout << "m3 = " << m3 << endl;
	cout << "m4 = " << m4 << endl;
	cout << "m5 = " << m5 << endl;

	waitKey();
}
