/*
 * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */
/*
 * This source code has been modified by Alex Epstein 
 * 	 Added features:
 * 		-By utilizing the area of the detected face we can look to recognize only the closest face
 * 		-An array that can store the names of the users so the output will predict it is jack rather than label 4
 * 		-Utilzing all methods of facial recognition we can determine if the prediction is accurate before proceeding 
 * 		-By using the LBPH facial recognition method I implemented adaptive training of the model based on confidence of prediction
 * 		-Detection and recognition all occur in one file to optimze the process rather than having to run one script after another through php
 *		-The output of the recognition is stored in a text file with the time along with other information about whether the model has been updated or not for use in further application
 */
 
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <ctime>
#include <fstream>
#include <sstream>
using namespace cv;
using namespace cv::face;
using namespace std;

// Function Headers
void detect(Mat frame);
void recognize();



//Global Variables
  bool speedTest = false;
  string listOfPeople[11] = {"no", //Person 0
					"ddf", //Person 1
					"fd", //Person 2
					"fdd", //Person 3
					"Ellen Degeneres", //Person 4
					"dfds" , //Person 5
					"person", //Person 6
					"pep", //Person 7
					"pp" , //Person 8
					"fd", //Person 9
					 "hello" //Person 10
				};
	string face_cascade_name = "/home/epsteina/Desktop/Data/facetrack-training.xml";
	CascadeClassifier face_cascade;
	int predictedLabel = -1;
	double confidence = 0.0;
	int tryrec = 0;
	int found = -1;
	clock_t begin;
    clock_t dstart;
    clock_t dend;
    clock_t start;
    clock_t end;
    clock_t g;
    clock_t h;
	clock_t s;
	clock_t j;
	clock_t n;
	double fishersecs;
	double secs;
	double LBPHsecs;
				

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(Error::StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}
int main(int argc, const char *argv[]) {
if (speedTest){
    begin = clock();
    dstart = clock();
}
while (tryrec <= 3 && found ==-1){
    // This would be the code used to grab the image 
    //* Get a handle to the Video device:
    
    VideoCapture cap(0);
    
    // Check if we can use this device at all:
    if(!cap.isOpened()) {
        cerr << "Capture Device ID cannot be opened." << endl;
        return -1;
    }
    // Holds the current frame from the Video device:
    Mat frame;
    
        cap >> frame;
	
	 cap.release();
	 //Attempt to load in the facial cascade that is used in detecting faces
	if (!face_cascade.load(face_cascade_name)){
        printf("--(!)Error loading\n");
        return (-1);
    }
    // Read the image file of ellen degeneres
   // Mat frame = imread("/home/epsteina/Desktop/ellen.jpg");
    
	 
	detect(frame);
}
	if (speedTest)
	{
	dend = clock();
	
	double delapsed_secs = double(dend - dstart) / CLOCKS_PER_SEC; 
    }
	//cout << "The detection took " << delapsed_secs << " seconds." << endl;
	recognize();
	if (speedTest)
	{
    end = clock();
	
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    }
	//Use this output to determine if any changes made in the code are actually optimizng the process
	//outfile << "The detection and recognition took " << elapsed_secs << " seconds."<<endl;
	
	/*
	 * To count elapsed time use this code
	 * clock_t begin = clock(); //this line will begin the clock
	 * 
	 * (Code you want to time goes here)
	 * 
	 * clock_t end = clock(); //this line will end the clock
	 * double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC; //this will set elapsed_secs to amount of passed seconds
	 * */
   
    return 0;
}
void recognize()
{
	if (speedTest)
	{
	 start = clock();
    }
    // Get the path to your CSV.
    string fn_csv = string("/home/epsteina/Desktop/Data/Faces.txt");
    // These vectors hold the images and corresponding labels.
    vector<Mat> images;
    vector<int> labels;
    // Read in the data. This can fail if no valid
    // input filename is given.
    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    // Quit if there are not enough images for this demo.
    if(images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(Error::StsError, error_message);
    }
    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size:
    int height = images[0].rows;
    // The following lines simply get the last images from
    // your dataset and remove it from the vector. This is
    // done, so that the training data (which we learn the
    // cv::BasicFaceRecognizer on) and the test data we test
    // the model with, do not overlap.
    Mat testSample = imread("/home/epsteina/Desktop/Data/tempface.png", CV_LOAD_IMAGE_GRAYSCALE);
    images.pop_back();
    labels.pop_back();
    
	
					
   // The following lines create an LBPH model for
    // face recognition and train it with the images and
    // labels read from the given CSV file.
    //
    // The LBPHFaceRecognizer uses Extended Local Binary Patterns
    // (it's probably configurable with other operators at a later
    // point), and has the following default values
    //
    //      radius = 1
    //      neighbors = 8
    //      grid_x = 8
    //      grid_y = 8
    //
    // So if you want a LBPH FaceRecognizer using a radius of
    // 2 and 16 neighbors, call the factory method with:
    //
    //      cv::createLBPHFaceRecognizer(2, 16);
    //
    // And if you want a threshold (e.g. 123.0) call it with its default values:
    //
    // 
    if (speedTest)
	{
		g = clock();
	}
    Ptr<LBPHFaceRecognizer> model = createLBPHFaceRecognizer(1,8,8,8,650);
    //model->train(images, labels);
    //model->save("/home/epsteina/Desktop/Data/LBPHFaceRec.xml");
    model->load("/home/epsteina/Desktop/Data/LBPHFaceRec.xml");
    model->predict(testSample, predictedLabel, confidence);
    string personsname = listOfPeople[predictedLabel]; //replace label with a name
	if (speedTest)
	{
		h = clock(); //this line will end the clock
	
	LBPHsecs = double(h - g) / CLOCKS_PER_SEC;
}
	
    //
    //Get the current time and translate it to coordinated universal time
     
     // current date/time based on current system
	time_t now = time(0);
   
   // convert now to string form
	char* dt = ctime(&now);
    
    // convert now to tm struct for UTC
   tm *gmtm = gmtime(&now);
   dt = asctime(gmtm);
   if (speedTest)
	{
    s = clock();
	}
    Ptr<FaceRecognizer> Fishermodel = createFisherFaceRecognizer(0,20000.0);
    Fishermodel->train(images, labels);
	int predicted = Fishermodel->predict(testSample);
	string Fisherperson = listOfPeople[predicted];
   	if (speedTest)
	{
   	j = clock(); //this line will end the clock
	fishersecs = double(j - s) / CLOCKS_PER_SEC;
    }  
	
    
    /*
    clock_t q = clock();
    Ptr<FaceRecognizer> EigenModel =  createEigenFaceRecognizer(10,20000.0);
    EigenModel->train(images, labels);
    int predictedcheck = EigenModel->predict(testSample);
    string Eigenperson = listOfPeople[predictedcheck];
    clock_t m = clock(); //this line will end the clock
	double Eigensecs = double(m - q) / CLOCKS_PER_SEC;
    */
	
    //
    // To get the confidence of a prediction call the model with:
    //
    //      int predictedLabel = -1;
    //      double confidence = 0.0;
    //      model->predict(testSample, predictedLabel, confidence);
    //
    cout << endl;
    cout << endl;
    //cout << "Predicted person through Fisher " << Fisherperson <<  endl;
    //cout << "Predicted person through Eigen " << Eigenperson << endl;
    //cout << "Predicted person through LBPH " << personsname  <<endl;
    cout << endl;
    cout << endl;
   //
   
	ofstream outfile;
	outfile.open("/home/epsteina/Desktop/Data/Recognition.txt", ios::out | ios::trunc );
   //
   outfile << "Confidence level of reading is " << confidence <<endl;
   //if ((predictedLabel == predicted) || (predictedLabel == predictedcheck)){
   if (predictedLabel == predicted){
	 if (confidence >= 70.0)
    {
		vector<Mat> upd; 
		vector<int> src;
		upd.push_back(imread("/home/epsteina/Desktop/Data/tempface.png", 0));
		src.push_back(predictedLabel);
		model->update(upd,src);
		model->save("/home/epsteina/Desktop/Data/LBPHFaceRec.xml");
		cout << personsname << endl;
		outfile << "Predicted name: " << personsname << " with a confidence of: " << confidence << endl;
		outfile << endl;
		outfile << dt << endl;
		outfile<< endl;
		outfile << "Facial recognition model has been updated for " << personsname <<endl;
		
	} 
	else if (confidence >=55 || confidence == 0)
	{
		if (confidence != 0){
		cout << personsname << endl;
		outfile<< "Predicted name: " << personsname << " with a confidence of: " << confidence << endl;
	}
		else
		{
		cout << personsname << endl;
		outfile<< "Predicted name: " << personsname << endl ;
		}
		outfile << endl;
		outfile << dt << endl;
		outfile << endl;
		outfile << "Facial recognition model will not be updated for " << personsname << " due to the lack of useable data. "  <<endl;
	}
	else
	{
		cout << "Guest" << endl;
		outfile << "Predicted name: Guest User" <<  endl;
		outfile << endl;
		outfile << dt << endl;
	}
}
else
{
	cout << "Guest" << endl;
	outfile << "Predicted name: Guest User" <<  endl;
	outfile << endl;
	outfile << dt << endl;
}

	if (speedTest)
	{
	n = clock(); //this line will end the clock
	secs = double(n - start) / CLOCKS_PER_SEC;
	outfile << endl;
	outfile << endl;
	
	outfile << "The LBPH Face recognition took " << LBPHsecs << " seconds."<<endl;
	outfile << "The Fisher Face recognition took " << fishersecs << " seconds."<<endl;
	//cout << "The Eigen Face recognition took " << Eigensecs << " seconds."<<endl;
	outfile<< endl;
	outfile << endl;
	outfile << "The total facial recognition took " << secs << " seconds."<<endl;
	}
	outfile<< endl;
	outfile<< endl;
	outfile.close();
}

void detect(Mat frame)
{
    std::vector<Rect> faces;
    Mat frame_gray;
    Mat crop;
    Mat res;
    Mat gray;
    string text;
    stringstream sstm;

    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    // Detect faces
    face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    // Set Region of Interest
    cv::Rect roi_b;
    cv::Rect roi_c;



    size_t ic = 0; // ic is index of current element
    int ac = 0; // ac is area of current element

    size_t ib = 0; // ib is index of biggest element
    int ab = 0; // ab is area of biggest element
	
if (faces.size() > 1){
    found = 1;
    for (ic = 0; ic < faces.size(); ic++) // Iterate through all current elements (detected faces)

    {
        roi_c.x = faces[ic].x;
        roi_c.y = faces[ic].y;
        roi_c.width = (faces[ic].width);
        roi_c.height = (faces[ic].height);

        ac = roi_c.width * roi_c.height; // Get the area of current element (detected face)

        roi_b.x = faces[ib].x;
        roi_b.y = faces[ib].y;
        roi_b.width = (faces[ib].width);
        roi_b.height = (faces[ib].height);

        ab = roi_b.width * roi_b.height; // Get the area of biggest element, at beginning it is same as "current" element

		//the next few lines attempt to extract the face that is closest to the camera
        if (ac > ab)     
        {
            ib = ic;
            roi_b.x = faces[ib].x;
            roi_b.y = faces[ib].y;
            roi_b.width = (faces[ib].width);
            roi_b.height = (faces[ib].height);
        }
	crop = frame(roi_b);
        resize(crop, res, Size(92, 112), 0, 0, INTER_NEAREST); // This will be needed later while saving images (size must be 92,112 to work with facial recognition algo)
        cvtColor(res, gray, CV_BGR2GRAY); // Convert cropped image to Grayscale

        // Form a filename
        string filename = "tempface.png";
        imwrite(filename, gray);
	}}
else if (faces.size()==1){
	found = 1;
	roi_b.x = faces[0].x;
        roi_b.y = faces[0].y;
        roi_b.width = (faces[0].width);
        roi_b.height = (faces[0].height);
	crop = frame(roi_b);
        resize(crop, res, Size(92, 112), 0, 0, INTER_NEAREST); // This will be needed later while saving images (size must be 92,112 to work with facial recognition algo)
        cvtColor(res, gray, CV_BGR2GRAY); // Convert cropped image to Grayscale

        // Form a filename
        string filename = "tempface.png";
        imwrite(filename, gray);
	}
else
{
	tryrec = tryrec + 1;
}

       
        
}
