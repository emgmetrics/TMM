
/**************************************************************************
TMM - Template Match with Mask

Detectors: 
	"FAST" – FastFeatureDetector
	"STAR" – StarFeatureDetector
	"SIFT" – SIFT (nonfree module)
	"SURF" – SURF (nonfree module)
	"ORB" – ORB
	"MSER" – MSER
	"GFTT" – GoodFeaturesToTrackDetector
	"HARRIS" – GoodFeaturesToTrackDetector with Harris detector enabled
	"Dense" – DenseFeatureDetector
	"SimpleBlob" – SimpleBlobDetector
	"BRISK" - BRISK
	Grid{Detector} 
	Pyramid{Detector} 
	Dynamic{Detector} 

Descriptors: 
	float descriptors: 
	"SIFT"
	"SURF"

	uchar descriptors:
	"ORB"
	"BRIEF"

Matchers: 
	For float descriptor:
	"FlannBased"
	"BruteForce"
	"BruteForce-L1"

	For uchar descriptor:
	"BruteForce-Hamming"
	"BruteForce-HammingLUT"

**************************************************************************/


#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui_c.h"

#include "Matcher.h"
#include "Extractor.h"
#include "Timer.h"

#ifdef _OPENCV_NON_FREE_
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "features2d.cpp"
#endif


using namespace std;
using namespace cv;

void readme();
void FeatureMatchingMethod( Mat& img_1, Mat& img_2, std::vector<Point> *matchLocations, int max_num_of_matches_to_find );
void ExactMatchingMethod( Mat& img, Mat& templ, std::vector<Point> *matchLocations );
int  MatchLocations (std::vector<Point> *locations1, std::vector<Point> *locations2);

#define _WAIT_KEY_TIMEOUT 0


void readme()
{
	std::cout << " Usage: ./TMM.exe <images data folder name> <show-images=0/1>" << std::endl;
}


Mat global_mask_img;
bool capturing_mask = false;


void mouseHandler(int _event, int x, int y, int flags, void *param)
{

	static bool button_down = false;
	static Point initial_point;
	Mat *img = (Mat *)param;
	static	Mat mask_img, tmp_img;
	//static bool capturing_mask = false;

	//x = x*((img->cols+100)/img->cols);
	//y = y*((img->rows+100)/img->rows);
	RECT win_rec;
	HWND win_handle = FindWindow(0, L"Mask");
	GetClientRect(
	  win_handle,
	  &win_rec);

	x = img->cols<win_rec.right-1?x*((float)win_rec.right/(float)img->cols):x;
	y = img->rows<win_rec.bottom-1?y*((float)win_rec.bottom/(float)img->rows):y;
			
	x = x>5000?0:x;
	y = y>5000?0:y;

	x = x>=global_mask_img.cols?global_mask_img.cols-1:x;
	y = y>=global_mask_img.rows?global_mask_img.rows-1:y;	

	switch(_event) {
	case EVENT_LBUTTONDOWN:		//left button press
		//cout << "On CV_EVENT_LBUTTONDOWN: (" << x << ", " << y << ")"<< endl;
		initial_point.x = x;
		initial_point.y = y;
		button_down = true;
		if (!capturing_mask)
		{
			global_mask_img.create(img->rows, img->cols, CV_8UC1);
			global_mask_img = 1;
		}
		break;

	case EVENT_LBUTTONUP:		//left mouse button release
		{
			//cout << "On CV_EVENT_LBUTTONUP: (" << x << ", " << y << ")"<< endl;
			mask_img = tmp_img.clone();
			button_down = false;


			x = x>5000?0:x;
			y = y>5000?0:y;

			x = x>=global_mask_img.cols?global_mask_img.cols-1:x;
			y = y>=global_mask_img.rows?global_mask_img.rows-1:y;	

			int tmp_x, tmp_y;
			if (x<initial_point.x)
			{
				tmp_x = x;
				x = initial_point.x;
				initial_point.x = tmp_x;
			}

			if (y<initial_point.y)
			{
				tmp_y = y;
				y = initial_point.y;
				initial_point.y = tmp_y;
			}

			for (int i=initial_point.x; i<=x; i++)
				for (int j=initial_point.y; j<=y; j++)
					global_mask_img.data[i + j*global_mask_img.cols] = 0;

		}
		break;

	case EVENT_RBUTTONUP:	//right mouse button release
		{
			//cout << "On CV_EVENT_RBUTTONUP" << endl;
			capturing_mask = false;
			for (int i=0; i<img->cols; i++)
				for (int j=0; j<img->rows; j++)
					tmp_img.data[i + j*img->cols] = global_mask_img.data[i + j*img->cols]?img->data[i + j*img->cols]:0;
			imshow( "Mask", tmp_img );
		}
		break;

	case EVENT_MOUSEMOVE:
		if (button_down && img)
		{
			if (!capturing_mask)
			{
				mask_img = img->clone();
				capturing_mask = true;
			}
			tmp_img = mask_img.clone();

			rectangle(tmp_img, initial_point, Point(x, y), Scalar(100,100,100), CV_FILLED);
  		    imshow( "Mask", tmp_img );
		}
		break;
		//cvReleaseImage(&temp);
	}
}

int main( int argc, char** argv )
{

  if( argc != 3 )
  { readme(); return -1; }

  _IMG_SHOW_ = atoi(argv[2]);

  save_files_dir_name = argv[1];
  std::string screen_img_name = save_files_dir_name + "\\screen.PNG";
  std::string template_img_name;

  Mat img_2 = imread( screen_img_name.c_str(), cv::ImreadModes::IMREAD_GRAYSCALE);
  if( !img_2.data )
  { 
	  std::cout<< " --(!) Error reading screen image " << screen_img_name.c_str() << std::endl;
	  return -1;
  }

  for (int i=0; i<12; i++)
  {
	  char img_name[10];
	  sprintf_s (img_name, "\\t%0.2d.PNG", i);
	  template_img_name = save_files_dir_name + img_name;

	  Mat img_1 = imread( template_img_name.c_str(), cv::ImreadModes::IMREAD_GRAYSCALE);
	  if( !img_1.data )
	  { 
		  continue;
	  }

	  cout << "Template image name: " << template_img_name.c_str() << endl;
  
	  global_mask_img.create(img_1.rows, img_1.cols, CV_8UC1);
	  global_mask_img = 1;

	  namedWindow("Mask", WINDOW_AUTOSIZE);
	  imshow("Mask", img_1);
	  setMouseCallback( "Mask", mouseHandler, &img_1 );
	  int key_pressed = waitKey(_WAIT_KEY_TIMEOUT);
	  capturing_mask = false;
	  switch(key_pressed)
	  {
			case 24: /*Ctrl-x - exit*/
				destroyAllWindows();
				return 0;
			case 3:  /*Ctrl-r - continue with next image*/
  				destroyAllWindows();
				continue;
			case 18: /*Ctrl-r - return to previous image*/
				if (i>0)
					i-=2;
  				destroyAllWindows();
				continue;			
			default:
				break;
	  }

	  float number_of_match_locations;
	  std::vector<Point> exactMatchLocations;
	  std::vector<Point> featureMatchLocations;

	  ExactMatchingMethod( img_2, img_1, &exactMatchLocations );

#if 0
	  FeatureMatchingMethod( img_1, img_2, &featureMatchLocations, exactMatchLocations.size() );
#else
	  MatchesDescriptionVector matchesDescriptionVect;
	  matchesDescriptionVect = TMMFeatureMatchTemplate(img_1, img_2, global_mask_img, 0.9, 20 /*exactMatchLocations.size()*/);

	  // convert MatchesDescriptionVector to std::vector<Point>.
	  for (size_t i=0; i<matchesDescriptionVect.size(); i++)
	  {
	    featureMatchLocations.push_back(Point(matchesDescriptionVect.at(i).first.x, matchesDescriptionVect.at(i).first.y));
	  }
#endif
	  number_of_match_locations = MatchLocations(&exactMatchLocations, &featureMatchLocations);

	  float matching_score = ((float)number_of_match_locations/(float)exactMatchLocations.size())*100.0f;
	  cout << "Matching score: " << matching_score << "%" << endl;
	  cout << endl;
 

	  if (_IMG_SHOW_ /*&& matching_score!=100.0f*/)
	  {
  		key_pressed = waitKey(_WAIT_KEY_TIMEOUT);
		switch(key_pressed)
		{
			case 24:
				destroyAllWindows();
				return 0;
			case 3:
				i--;
			default:
				break;
		}
	  }
  	  destroyAllWindows();
  }
  return 0;
}

int MatchLocations (std::vector<Point> *locations1, std::vector<Point> *locations2)
{
	int match_loc = 0;
	double min_dist = 15.0;
	double min_norm;

	for (size_t i = 0; i < locations1->size(); i++)
		for (size_t j = 0; j<locations2->size(); j++)
		{
			min_norm = std::sqrt((double)std::norm(std::complex<int>((locations1->at(i).x - locations2->at(j).x), (locations1->at(i).y - locations2->at(j).y))));

			if( min_norm <= min_dist)
			{
				min_dist = min_norm;
				//(abs(locations1->at(i).x - locations2->at(j).x) < 20 &&
				//abs(locations1->at(i).y - locations2->at(j).y) < 20 )
				match_loc++;
			}
		}
	return match_loc;
}

void ExactMatchingMethod( Mat& img, Mat& templ, std::vector<Point> *matchLocations )
{
	Timer tmr;

	Mat result;
	int match_method = CV_TM_CCOEFF_NORMED; //CV_TM_SQDIFF_NORMED; //CV_TM_CCOEFF_NORMED; //CV_TM_CCORR_NORMED;
	float TRESHOLD = 0.03;


	/// Source image to display
	Mat img_display;
	img.copyTo( img_display );

	/// Create the result matrix
	int result_cols =  img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;

	result.create( result_cols, result_rows, CV_32FC1 );

	tmr.Start();

	/// Do the Matching and Normalize
	matchTemplate( img, templ, result, match_method );
	//normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );

	/// Localizing the best match with minMaxLoc
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;

	minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc );

	//rectangle( img_display, maxLoc, Point( maxLoc.x + templ.cols , maxLoc.y + templ.rows ), Scalar::all(0), 2, 8, 0 );

    const float* src = (const float*)result.data;
    size_t step = result.step/sizeof(src[0]);
    float min_val = src[0], max_val = min_val;
    int min_loc = 0, max_loc = 0;
    int x, loc = 0;
    //Size size = getContinuousSize( srcmat );
    Size size = Size(result.cols, result.rows);

    for( ; size.height--; src += step, loc += size.width )
    {
        for( x = 0; x < size.width; x++ )
        {
            float val = src[x];
			if( std::abs(val - maxVal) < TRESHOLD )
            {
              max_val = maxVal;
              max_loc = loc + x;
			  if( max_loc )
			  {
					if( max_loc >= 0 )
					{
						maxLoc.y = max_loc/result.cols;
						maxLoc.x = max_loc - maxLoc.y*result.cols;
					}
					else
						maxLoc.x = maxLoc.y = -1;
			  }
			  if( match_method  == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED )
				{ matchLoc = minLoc; }
			  else
				{ matchLoc = maxLoc; }

			  /// Show me what you got
			  rectangle( img_display, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar::all(0), 2, 8, 0 );
			  matchLocations->push_back(matchLoc);

			}
        }
    }

	tmr.Stop();
	cout << "ExactMatchingMethod elapse time: " << tmr.GetTime() << endl;
    cout << "Exact Matching found: " << matchLocations->size() << " templates in screen image." << endl;

	/// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better


	if (_IMG_SHOW_)
	{
		namedWindow( "ExactMatchingMethod Window", WINDOW_AUTOSIZE );
		imshow( "ExactMatchingMethod Window", img_display );
	}
}

void FeatureMatchingMethod( Mat& img_1, Mat& img_2, std::vector<Point> *matchLocations, int max_num_of_matches_to_find )
{
	Timer tmr;

#ifdef _OPENCV_NON_FREE_
	bool res = cv::initModule_nonfree();
#endif

	std::vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;
	Mat moments_descriptors_1, moments_descriptors_2;


	bool _MASK_TEMPLATE_ = true;
	Mat mask(global_mask_img);

	tmr.Start();

	//
	// Step 1: Detect Template keypoints and calculate TMM Template descriptor.
	//
	TMMDescriptorExtractor tmm_extractor;
	tmm_extractor.computeTemplateDescriptor(img_1, keypoints_1, descriptors_1, mask);


	//
	// Step 2: Detect screen image keypoints using FAST (FastFeatureDetector) detector:
	//
#if 1
	FastFeatureDetector *fast_detector = FastFeatureDetector::create(FAST_THRESHOLD);
	fast_detector->detect( img_2, keypoints_2 );
#else
	Ptr<FeatureDetector> detector_tmm = FeatureDetector::create( "MSER" );
	detector_tmm->detect( img_2, keypoints_2 );
#endif

	//
	// Step 3: Calculate Screen Image descriptors (feature vectors) by using FAST keypoints
	// and TMM Descriptor Extractor.
	//
	tmm_extractor.compute(img_2, keypoints_2, descriptors_2);

	tmr.Stop();
	cout << "Descriptors calculation elapse time: " << tmr.GetTime() << "sec." << endl;
	cout << "Number of keypoints on Template: " << keypoints_1.size() << endl;
	cout << "Number of keypoints on Screen  : " << keypoints_2.size() << endl;
	cout << "Angles threshold: " << _DESCRIPTOR_THRESHOLD_ANGLE_ << "/" << DESCRIPTOR_THRESHOLD_ANGLE << endl;


	//
	// Step 4: Matching descriptor-vectors using TMM matcher.
	//

	tmr.Start();

#if 0
	std::vector< DMatch > good_matches;
	std::vector< DMatch > matches_2;
	int max_dist = 0; int min_dist = 20000;
	std::vector<KeyPoint> good_keypoints;

	//
	// Match descriptors by Angle and image mean.
	//

	int angle_dist, mean_dist;
	int descriptor_size = tmm_extractor.descriptorSize();

	for (size_t i = 0; i < keypoints_2.size(); i++)
	{
		uchar* size_i = descriptors_2.ptr((int)i);
		uchar* size_t = descriptors_1.ptr((int)0);
		angle_dist = 0;
		for (int j=0; j<descriptor_size-1; j++)
		{
			uchar xor_bit_diff = (size_i[j] ^ size_t[j]);
			uchar bit_diff = ~(size_i[j] ^ size_t[j]);//size_i[j] & size_t[j];
			if (bit_diff != 0)
			{
				for (int k=0; k<8; k++)
					angle_dist += (bit_diff & 1<<k) != 0;
			}
		}

		matches_2.push_back(DMatch(i, 0, (float)angle_dist));
		mean_dist = abs(size_i[descriptor_size-1]-size_t[descriptor_size-1]);

		//-- Quick calculation of max and min distances between keypoints
		if( angle_dist < min_dist ) min_dist = angle_dist;
		if( angle_dist > max_dist ) max_dist = angle_dist;

		if( angle_dist > ANGLE_DIST_THRESHOLD && mean_dist < MEAN_DIST_THRESHOLD )
		{ 
			if (good_keypoints.empty())
			{
				good_matches.push_back(matches_2[i]);
				good_keypoints.push_back( keypoints_2.at(matches_2[i].queryIdx) );
			}
			else
			{
				bool _insert = false;
				for (int k=0; k<good_keypoints.size(); k++)
				{
					float _norm = std::sqrt(std::pow(good_keypoints.at(k).pt.x - keypoints_2.at(i).pt.x,2) +
											std::pow(good_keypoints.at(k).pt.y - keypoints_2.at(i).pt.y,2));

					if (_norm > 10.0f)
					{
						_insert = true;
					}
					else
					{
						if ( matches_2[i].distance > good_matches.at(k).distance )
						{
							good_matches.erase(good_matches.begin()+k);
							good_keypoints.erase(good_keypoints.begin()+k);
					
							good_matches.push_back( matches_2[i]);
							good_keypoints.push_back( keypoints_2.at(matches_2[i].queryIdx) );

						}
						_insert = false;
						break;
					}
				}
				if(_insert)
				{
					good_matches.push_back( matches_2[i]);
					good_keypoints.push_back( keypoints_2.at(matches_2[i].queryIdx) );
				}
			}
		}
	}
	cout << "-- Max dist : " << max_dist << endl;
	cout << "-- Min dist : " << min_dist << endl;


	//
	// Filter current calculated matches by descriptors Moments.
	//
	tmm_extractor.computeKeypointsMomentsDescriptors(img_2, good_keypoints, moments_descriptors_2, mask);
	
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	std::vector< vector<DMatch> > end_matches;
	//matcher->knnMatch(moments_descriptors_1, moments_descriptors_2, end_matches, 10/*max_num_of_matches_to_find*/ );
	matcher->radiusMatch(moments_descriptors_1, moments_descriptors_2, end_matches, 500.0f);

	thresholdMomentsDescriptors(end_matches[0]);

#else

	//
	// Step 5: Filter current calculated matches by descriptors Moments.
	//
	
	// Calculate Template Moments-descriptor.
	tmm_extractor.computeTemplateMomentsDescriptor (img_1, keypoints_1, moments_descriptors_1);

	std::vector< vector<DMatch> > good_matches;
	std::vector< vector<DMatch> > end_matches;
	std::vector< KeyPoint > good_keypoints;
	
	// match descriptors by angles and mean descriptors.
	TMMatcher tmm_matcher(&keypoints_2);
	tmm_matcher.knnMatch(descriptors_1, descriptors_2, good_matches, 10);

	// For each matched screen descriptor, calculate Moments-descriptor.
	for (size_t match_indx=0; match_indx<good_matches[0].size(); match_indx++)
	{
		good_keypoints.push_back( keypoints_2.at(good_matches[0].at(match_indx).queryIdx) );
	}
	tmm_extractor.computeKeypointsMomentsDescriptors(img_2, good_keypoints, moments_descriptors_2, mask);


	//
	// Step 6: Match moments-descriptors using BruteForce matcher.
	//         and eturn only the matches under constrain.
	//
	Ptr<DescriptorMatcher> bf_matcher = DescriptorMatcher::create("BruteForce");
#if 0
	// Return only the best N matches.
	int N = 10/*max_num_of_matches_to_find*/;
	bf_matcher->knnMatch(moments_descriptors_1, moments_descriptors_2, end_matches, N);
#else
	// Return only the matches under threshold constrain.
	bf_matcher->radiusMatch(moments_descriptors_1, moments_descriptors_2, end_matches, 500.0f);
	thresholdMomentsDescriptors(end_matches[0], 0.9);
#endif


	tmr.Stop();
	cout << "Feature template matching elapse time: " << tmr.GetTime() << " sec." << endl;


#endif

	//
	// Step7: Calculate the Template Matches image position on the screen image.
	//
	Point distance = tmm_extractor.getDist();
	for (size_t x=0; x<end_matches.at(0).size(); x++)
	{
	  Point ptLoc;
	  ptLoc = good_keypoints.at(end_matches.at(0).at(x).trainIdx).pt;
	  ptLoc.x -= distance.x;
	  ptLoc.y -= distance.y;
	  matchLocations->push_back( ptLoc );
	}

	//
	// show and save detected (drawn) keypoints.
	//
  	bool ret_val = false;

	Mat img_keypoints_1; Mat img_keypoints_2;
	drawKeypoints( img_1, keypoints_1, img_keypoints_1, Scalar(200), DrawMatchesFlags::DEFAULT );
	drawKeypoints( img_2, keypoints_2, img_keypoints_2, Scalar(200), DrawMatchesFlags::DEFAULT );
	ret_val = imwrite(save_files_dir_name+"\\template_keypoints.png",img_keypoints_1);
	ret_val = imwrite(save_files_dir_name+"\\screen_keypoints.png",img_keypoints_2);

	//
	// show descriptors matches before moments calculation
	//
	Mat img_matches_1;
	drawMatches( img_2, keypoints_2, img_1, keypoints_1,
			   good_matches[0], img_matches_1, Scalar(200), Scalar(200),
			   vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	ret_val = imwrite(save_files_dir_name+"\\good_matches.png",img_matches_1);

	//
	// show descriptors matches after moments calculation
	//
	Mat img_matches_2;
	drawMatches( img_1, keypoints_1, img_2, good_keypoints,
			   end_matches, img_matches_2, Scalar(200), Scalar(200),
			   vector<vector<char>>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
	ret_val = imwrite(save_files_dir_name+"\\moments_matches.png",img_matches_2);
	

	if (_IMG_SHOW_)
	{  
		imshow("Keypoints 1", img_keypoints_1 );
		imshow("Keypoints 2", img_keypoints_2 );
		imshow( "Good Matches", img_matches_1 );
		imshow( "Moments Matches", img_matches_2 );
	}
}
