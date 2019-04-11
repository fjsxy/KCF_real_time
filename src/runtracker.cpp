#include <iostream>
#include <fstream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "kcftracker.hpp"

#define DATA_PATH "/home/fjsxy/KCFcpp_real_time/build/data/"
#define FACAL 810
#define REAL_LENGTH 34
//#define REAL_TIME
//#define NO_REAL_TIME

int times = 0;

using namespace std;
using namespace cv;

Point2f a[2];

void on_mouse(int event,int x,int y,int flags,void* ustc)
{
    switch (event){
        case cv::EVENT_LBUTTONDOWN:{
            a[times].x = x;a[times].y = y;
            times++;
        }
            break;
        default:;
    }
}

int main(int argc, char* argv[]){

	if (argc > 5) return -1;

	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool SILENT = false;
	bool LAB = false;

	for(int i = 0; i < argc; i++){
		if ( strcmp (argv[i], "hog") == 0 )
			HOG = true;
		if ( strcmp (argv[i], "fixed_window") == 0 )
			FIXEDWINDOW = true;
		if ( strcmp (argv[i], "singlescale") == 0 )
			MULTISCALE = false;
		if ( strcmp (argv[i], "show") == 0 )
			SILENT = false;
		if ( strcmp (argv[i], "lab") == 0 ){
			LAB = true;
			HOG = true;
		}
		if ( strcmp (argv[i], "gray") == 0 )
			HOG = false;
	}

    // open camera,use capture to get frame
    VideoCapture capture(0);
    if(!capture.isOpened()){
        cout << "capture failed." <<endl;
        return -1;
    }


	// Create KCFTracker object
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

	// Frame readed
	Mat frame,frame_gray;

	// Tracker results
	Rect result;

    //program end flag
    bool stop = false;

  	// Read groundtruth like a dumb
  	float x1,y1,x2,y2,x3,y3,x4,y4;

	// Using min and max of X and Y for groundtruth rectangle
	float xMin,yMin,width,height;

    float length;
    float depth;

	// Write Results
	ofstream resultsFile;
	string resultsPath = DATA_PATH"output.txt";
	resultsFile.open(resultsPath);

	// Frame counter
	int nFrames = 0;

	while (!stop){

		// Read each frame from the capture
        capture >> frame;
        circle(frame, Point(320,240),2,Scalar(0,0,255),-1);

        cv::setMouseCallback("Image",on_mouse,&frame);

        if (times > 1) {

            // First frame, give the groundtruth to the tracker
            if (nFrames == 0) {

                // get rect two point
                x1 = a[0].x;y1 = a[0].y;x2 = a[1].x;y2 = a[1].y;
                x3 = x1;y3 = y2;x4 = x2;y4 = y1;
                xMin =  min(x1, min(x2, min(x3, x4)));
                yMin =  min(y1, min(y2, min(y3, y4)));
                width = max(x1, max(x2, max(x3, x4))) - xMin;
                height = max(y1, max(y2, max(y3, y4))) - yMin;

                tracker.init(Rect(xMin, yMin, width, height), frame);
                rectangle(frame, Point(xMin, yMin), Point(xMin + width, yMin + height), Scalar(0, 255, 255), 1, 8);
                circle(frame, Point(xMin + width / 2,yMin + height / 2),1,Scalar(255,0,0),-1);
                resultsFile << xMin << "," << yMin << "," << width << "," << height << endl;
                length = width ? height : width > height;
                depth = FACAL * REAL_LENGTH / length;
            }
                // Update
            else {
                result = tracker.update(frame);
                rectangle(frame, Point(result.x, result.y), Point(result.x + result.width, result.y + result.height),
                          Scalar(0, 255, 255), 1, 8);
                circle(frame, Point(result.x + result.width / 2,result.y + result.height / 2),1,Scalar(255,0,0),-1);
                resultsFile << result.x << "," << result.y << "," << result.width << "," << result.height << endl;
                length = result.width ? result.height : result.width > result.height;
                depth = FACAL * REAL_LENGTH / length;
            }
            int baseline;
            Size text_size = getTextSize(to_string(depth),FONT_HERSHEY_COMPLEX,1,2,&baseline);
            putText(frame,to_string(depth),Point(0,text_size.height),FONT_HERSHEY_COMPLEX,1,Scalar(0,0,0),2,8,0);

            nFrames++;
        }

		if (!SILENT){
			imshow("Image", frame);
			if (waitKey(30) > 0)
                stop = true;
		}
	}
	resultsFile.close();

//	listFile.close();

}
