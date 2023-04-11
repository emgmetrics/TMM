#include "Matcher.h"
#include "Extractor.h"
#include "Timer.h"

#include <fstream>
#include <iomanip>


#ifdef _DEMO_
#define LOG4CPLUS_INFO(logger, args) cout << args << endl
#else
#include <log4cplus/consoleappender.h>
#include <log4cplus/hierarchy.h>
using namespace log4cplus;
#endif

using namespace std;
using namespace cv;

std::string save_files_dir_name = ".";
bool _IMG_SHOW_ = false;

Ptr<FeatureDetector> CreateFeatureDetector (DetectorType detectorType);
Scalar getMSSIM( const Mat& i1, const Mat& i2, const Mat& mask);
double getPSNR(const Mat& I1, const Mat& I2);

float thresholdMomentsDescriptors (std::vector<DMatch> &dvector, double threshold)
{
	std::vector<DMatch>::iterator v=dvector.begin();
	v=dvector.begin();
	int count = 0;
	while(v!=dvector.end())
	{
			if ( ((*v).distance > FILTER_THRESHOLD_TOLERANCE) )
			{
				if (threshold == 1.0 || count > DEFAULT_NUM_OF_MATCHES_TO_FIND)
					v = dvector.erase(v, dvector.end());
				else
				{
					count++;
					v++;
				}
			}
			else
			{
				v++;
			}
	}
	return 0;
}

static bool compare_by_dist(const DMatch keypoint1, const DMatch keypoint2)
{
	return ( keypoint1.distance > keypoint2.distance );
}

void pixel_pixelMatch (
		const Mat& img_1,
		const Mat& img_2, 
		const std::vector<KeyPoint>& good_keypoints, 
		const std::vector<DMatch>& good_matches, 
		std::vector< vector<DMatch> >& end_matches,
		const Point& distance,
		const Mat& mask,
		double threshold)
{
	std::vector<DMatch> v_matches, v1_matches, v2_matches;

	for (size_t x=0; x<good_matches.size(); x++)
	{

		Point ptLoc;
		ptLoc = good_keypoints.at(x).pt;
		ptLoc.x -= distance.x;
		ptLoc.y -= distance.y;

		int R = 0, unmasked_px_count = 0;
		double cnt_px_percent = 0;

		int cx, ry;

		Rect roi(ptLoc.x, ptLoc.y, img_1.cols, img_1.rows);
		Mat img_2_roi = img_2(roi);
		
		for (ry=0; ry<img_1.rows; ry++)
		{
			for (cx=0; cx<img_1.cols; cx++)
			{
				if (mask.at<uchar>(ry, cx))// unmasked pixel!=0.
				{
					R += (img_2_roi.at<uchar>(ry,cx) ^ img_1.at<uchar>(ry, cx))==0?1:0;
					unmasked_px_count++;
				}
			}
		}
		cnt_px_percent = (double)R / (double)unmasked_px_count;
		if (cnt_px_percent > threshold - FILTER_THRESHOLD_TOLERANCE)
		{			
			DMatch match;
			match.trainIdx = x;
			match.queryIdx = 0;
			match.imgIdx = 0;
			match.distance = cnt_px_percent;
			if (match.distance > 1.0 - FILTER_THRESHOLD_TOLERANCE)
				v1_matches.push_back(match);
			else
				v2_matches.push_back(match);
		}
	}

	if (!v1_matches.empty())
	{
		std::sort(v1_matches.begin(), v1_matches.end(),compare_by_dist);
		v_matches.insert(v_matches.end(), v1_matches.begin(), v1_matches.end());
	}
	if (!v2_matches.empty())
	{
		std::sort(v2_matches.begin(), v2_matches.end(),compare_by_dist);
		if (v2_matches.size() > DEFAULT_NUM_OF_MATCHES_TO_FIND)
			v2_matches.erase(v2_matches.begin()+DEFAULT_NUM_OF_MATCHES_TO_FIND, v2_matches.end());
		v_matches.insert(v_matches.end(), v2_matches.begin(), v2_matches.end());
	}
	if (!v_matches.empty())
	{
		end_matches.push_back(v_matches);
	}
}


void mssimMatch (
		const Mat& img_1,
		const Mat& img_2, 
		const std::vector<KeyPoint>& good_keypoints, 
		const std::vector<DMatch>& good_matches, 
		std::vector< vector<DMatch> >& end_matches,
		const Point& distance,
		const Mat& mask,
		double threshold)
{
	int NB_SIZE = 0;
	std::vector<DMatch> v_matches, v1_matches, v2_matches;

	for (size_t x=0; x<good_matches.size(); x++)
	{
		Point ptLoc;
		ptLoc = good_keypoints.at(x).pt;
		ptLoc.x -= distance.x;
		ptLoc.y -= distance.y;

		for (int nx=-1*NB_SIZE; nx<=NB_SIZE; nx++)
		  for (int ny=-1*NB_SIZE; ny<=NB_SIZE; ny++)
		  {

			  Rect roi(ptLoc.x - nx, ptLoc.y - ny, img_1.cols, img_1.rows);
			  Mat img_2_roi = img_2(roi);
						  
			  Scalar mssim = getMSSIM(img_1, img_2_roi, mask);
			  if (mssim[0] > threshold - FILTER_THRESHOLD_TOLERANCE)
			  {			
					DMatch match;
					match.trainIdx = x;
					match.queryIdx = 0;
					match.imgIdx = 0;
					match.distance = mssim[0];
					if (match.distance > 1.0 - FILTER_THRESHOLD_TOLERANCE)
						v1_matches.push_back(match);
				   else
						v2_matches.push_back(match);			  }
		  }
	}

	if (!v1_matches.empty())
	{
		std::sort(v1_matches.begin(), v1_matches.end(),compare_by_dist);
		v_matches.insert(v_matches.end(), v1_matches.begin(), v1_matches.end());
	}
	if (!v2_matches.empty())
	{
		std::sort(v2_matches.begin(), v2_matches.end(),compare_by_dist);
		if (v2_matches.size() > DEFAULT_NUM_OF_MATCHES_TO_FIND)
			v2_matches.erase(v2_matches.begin()+DEFAULT_NUM_OF_MATCHES_TO_FIND, v2_matches.end());
		v_matches.insert(v_matches.end(), v2_matches.begin(), v2_matches.end());
	}
	if (!v_matches.empty())
	{
		end_matches.push_back(v_matches);
	}
}


MatchesDescriptionVector TMMFeatureMatchTemplate(
	Mat& img_1, // Template image.
	Mat& img_2, // Screen image.
	cv::Mat &mask,
	double threshold,
	int max_num_of_matches_to_find)
{

#ifndef _DEMO_
	static Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ClassificationEngine.FeatureMatchTemplate"));
#endif


	MatchesDescriptionVector matchLocations;

	Timer tmr;
	tmr.Start();

	Timer tmr1;
	tmr1.Start();

	std::vector<KeyPoint> keypoints_1, // Template image keypoints.
		                  keypoints_2; // Screen image keypoints.

	Mat descriptors_1, // Template descriptor.
		descriptors_2; // Descriptors for each screen image keypoint.

	Mat moments_descriptors_1, // Moments descriptor for Template image.
		moments_descriptors_2; // Moments descriptors for each screen image keypoint.

	//
	// Set Feature Template Match configuration thresholds.
	//
	FeatureTemplateMatchConfiguration::SetThresholds(threshold);

	//
	// Step 1: Detect Template keypoints and calculate Template descriptor.
	//
	TMMDescriptorExtractor tmm_extractor;
	tmm_extractor.setCurrentDetectorType(DetectorType::DETECTOR_FAST);
	LOG4CPLUS_INFO(logger, "Using DETECTOR_FAST.");
	tmm_extractor.computeTemplateDescriptor(img_1, keypoints_1, descriptors_1, mask);
	if (keypoints_1.empty())
	{
#ifdef _USE_ONLY_FAST_DETECTOR_
		LOG4CPLUS_ERROR(logger, "No Feature points found in Template image, return empty Positions vector.");
		return matchLocations;
#else
		tmm_extractor.setCurrentDetectorType(DetectorType::DETECTOR_GFTT_WITH_HARRIS);
		LOG4CPLUS_INFO(logger, "Using DETECTOR_GFTT_WITH_HARRIS.");
		tmm_extractor.computeTemplateDescriptor(img_1, keypoints_1, descriptors_1, mask);

		tmr1.Stop();
		LOG4CPLUS_INFO(logger, "Template Descriptor elapse time: " << tmr1.GetTime() << "sec.");

		if (keypoints_1.empty())
		{
			LOG4CPLUS_INFO(logger, "No Feature points found in Template image, return empty Positions vector.");
			return matchLocations;
		}
#endif
	}


	//
	// Step 2: Detect screen image keypoints using FAST (FastFeatureDetector) detector:
	//
#ifdef _USE_ONLY_FAST_DETECTOR_
	FastFeatureDetector fast_detector(FAST_THRESHOLD);
	fast_detector.detect( img_2, keypoints_2 );
#else
	tmr1.Reset();
	tmr1.Start();

	cv::FAST(img_2, keypoints_2, FAST_THRESHOLD);
	//Ptr<FeatureDetector> feature_detector = CreateFeatureDetector(tmm_extractor.getCurrentDetectorType());
	//feature_detector->detect( img_2, keypoints_2 );
	//feature_detector.release();

	tmr1.Stop();
	LOG4CPLUS_INFO(logger, "Screen feature detector elapse time: " << tmr1.GetTime() << "sec.");

#endif
	if (keypoints_2.empty())
	{
		LOG4CPLUS_INFO(logger, "No Feature points found in Screen image, return empty Positions vector.");
		return matchLocations;
	}

	//
	// Step 3: Calculate Screen Image descriptors (feature vectors) by using FAST keypoints
	// and TMM Descriptor Extractor.
	//
	
	tmr1.Reset();
	tmr1.Start();

	tmm_extractor.computeImpl(img_2, keypoints_2, descriptors_2);

	tmr1.Stop();
	LOG4CPLUS_INFO(logger, "Screen Image descriptors elapse time: " << tmr1.GetTime() << "sec.");

	tmr.Stop();

	LOG4CPLUS_INFO(logger, "Descriptors calculation elapse time: " << tmr.GetTime() << "sec.");
	LOG4CPLUS_INFO(logger, "Number of keypoints on Template: " << keypoints_1.size());
	LOG4CPLUS_INFO(logger, "Number of keypoints on Screen  : " << keypoints_2.size());
	LOG4CPLUS_INFO(logger, "Angles threshold: " << _DESCRIPTOR_THRESHOLD_ANGLE_ << "/" << DESCRIPTOR_THRESHOLD_ANGLE);

	//
	// Step 4: Matching descriptor-vectors using TMM matcher, and return all the good matches.
	//

	tmr.Start();

	std::vector< vector<DMatch> > good_matches;
	std::vector< vector<DMatch> > end_matches;
	std::vector< KeyPoint > good_keypoints;
	
	// match descriptors by angles and mean descriptors.
	TMMatcher tmm_matcher(&keypoints_2);
	tmm_matcher.knnMatch(descriptors_1, descriptors_2, good_matches, max_num_of_matches_to_find /*here all matches are returned*/);


	//
	// Step 5: Filter current calculated matches by descriptors Moments or by Pixel-Pixel
	//

	// get the relevant keypoints.
	for (size_t match_indx=0; match_indx<good_matches[0].size(); match_indx++)
	{
		good_keypoints.push_back( keypoints_2.at(good_matches[0].at(match_indx).queryIdx) );
	}

	if(FILTER_ALGORITHM == "Moments" || threshold == 1)
	{	
		// Calculate Template Moments-descriptor.
		tmm_extractor.computeTemplateMomentsDescriptor (img_1, keypoints_1, moments_descriptors_1);

		// For each relevant matched screen keypoint, calculate Moments-descriptor.
		tmm_extractor.computeKeypointsMomentsDescriptors(img_2, good_keypoints, moments_descriptors_2, mask);

		//
		// Step 6: Match moments-descriptors using BruteForce matcher.
		//         and return only the matches under constrain.
		//
		Ptr<DescriptorMatcher> bf_matcher = DescriptorMatcher::create("BruteForce");
		//BFMatcher bf_matcher(NORM_L1);

		// Return only the best N matches.
		max_num_of_matches_to_find = 100;
		bf_matcher->knnMatch(moments_descriptors_1, moments_descriptors_2, end_matches, max_num_of_matches_to_find);

		// Between all the best N matches, return only the matches under threshold constrain.	
		thresholdMomentsDescriptors(end_matches[0], threshold);
	}
	else if (FILTER_ALGORITHM == "MSSI")
		mssimMatch (img_1, img_2, good_keypoints, good_matches.at(0), end_matches, tmm_extractor.getDist(), mask, threshold);
	else
		pixel_pixelMatch (img_1, img_2, good_keypoints, good_matches.at(0), end_matches, tmm_extractor.getDist(), mask, threshold);

	tmr.Stop();
	LOG4CPLUS_INFO(logger, "Feature template matching elapse time: " << tmr.GetTime() << " sec");


	//
	// Step7: Calculate the Template Matches image position on the screen image.
	//
	LOG4CPLUS_INFO(logger, "Number of patches found: " << end_matches.size());

	Point distance = tmm_extractor.getDist();
	if (end_matches.size() > 0)
	{
		for (size_t x = 0; x < end_matches.at(0).size(); x++)
		{
			Point ptLoc;
			ptLoc = good_keypoints.at(end_matches.at(0).at(x).trainIdx).pt;
			ptLoc.x -= distance.x;
			ptLoc.y -= distance.y;

			matchLocations.push_back(std::pair<Point, float>(Point(ptLoc.x, ptLoc.y), end_matches.at(0).at(x).distance));
			LOG4CPLUS_INFO(logger, x << "  " << ptLoc.x << "  " << ptLoc.y << "  " << end_matches.at(0).at(x).distance << "  " << good_matches.at(0).at(end_matches.at(0).at(x).trainIdx).distance);

		}
	}


	//
	// show and save detected (drawn) keypoints.
	// print debug matches information.
	//
	if (/*FEATURE_MATCH_SAVE_IMAGES*/ 1)
	{
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


		//
		// Print matches information in diferent file (for testing output).
		//
		std::ofstream oregression_file;
		std::ifstream iregression_file("regression_values.out");
		if (iregression_file.is_open())
		{
			iregression_file.close();
			oregression_file.open("regression_values.out", std::ios_base::app);
			if(oregression_file.is_open())
			{
				for (size_t cnt=0; cnt<matchLocations.size(); cnt++)
					oregression_file << std::setw(3) << cnt << " " << std::setw(5) << matchLocations.at(cnt).first.x << " " << std::setw(5) << matchLocations.at(cnt).first.y << " " << std::setw(12) << std::setprecision(5) << matchLocations.at(cnt).second << "  " << std::setw(4) << good_matches.at(0).at(end_matches.at(0).at(cnt).trainIdx).distance << endl;

				oregression_file.close();
			}
		}
	}
	
	return matchLocations;
}

double getPSNR(const Mat& I1, const Mat& I2)
{
	 Mat s1;
	 absdiff(I1, I2, s1);       // |I1 - I2|
	 s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
	 s1 = s1.mul(s1);           // |I1 - I2|^2

	 Scalar s = sum(s1);         // sum elements per channel

	 double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

	 if( sse <= 1e-10) // for small values return zero
		 return 0;
	 else
	 {
		 double  mse =sse /(double)(I1.channels() * I1.total());
		 double psnr = 10.0*log10((255*255)/mse);
		 return psnr;
	 }
}

Scalar getMSSIM( const Mat& i1, const Mat& i2, const Mat& mask)
{
 const double C1 = 6.5025, C2 = 58.5225;
 /***************************** INITS **********************************/
 int d     = CV_32F;

 Mat I1, I2, Mask;
 i1.convertTo(I1, d);           // cannot calculate on one byte large values
 i2.convertTo(I2, d);
 mask.convertTo(Mask, d);

 if (!mask.empty())
 {
	 I1 = I1.mul(Mask);
	 I2 = I2.mul(Mask);
 }

 Mat I2_2   = I2.mul(I2);        // I2^2
 Mat I1_2   = I1.mul(I1);        // I1^2
 Mat I1_I2  = I1.mul(I2);        // I1 * I2

 /***********************PRELIMINARY COMPUTING ******************************/

 Mat mu1, mu2;   //
 cv::GaussianBlur(I1, mu1, Size(11, 11), 1.5);
 cv::GaussianBlur(I2, mu2, Size(11, 11), 1.5);

 Mat mu1_2   =   mu1.mul(mu1);
 Mat mu2_2   =   mu2.mul(mu2);
 Mat mu1_mu2 =   mu1.mul(mu2);

 Mat sigma1_2, sigma2_2, sigma12;

 cv::GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
 sigma1_2 -= mu1_2;

 cv::GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
 sigma2_2 -= mu2_2;

 cv::GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
 sigma12 -= mu1_mu2;

 ///////////////////////////////// FORMULA ////////////////////////////////
 Mat t1, t2, t3;

 t1 = 2 * mu1_mu2 + C1;
 t2 = 2 * sigma12 + C2;
 t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

 t1 = mu1_2 + mu2_2 + C1;
 t2 = sigma1_2 + sigma2_2 + C2;
 t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

 Mat ssim_map;
 divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

 Scalar mssim = mean( ssim_map ); // mssim = average of ssim map
 return mssim;
}

TMMatcher::TMMatcher( std::vector<KeyPoint>* trainKeypoints, bool crossCheck )
{
    m_crossCheck = crossCheck;
	m_trainKeypoints = trainKeypoints;
}

Ptr<DescriptorMatcher> TMMatcher::clone( bool emptyTrainData ) const
{
	TMMatcher* matcher = new TMMatcher(m_trainKeypoints, m_crossCheck);
    if( !emptyTrainData )
    {
        matcher->trainDescCollection.resize(trainDescCollection.size());
        std::transform( trainDescCollection.begin(), trainDescCollection.end(),
                        matcher->trainDescCollection.begin(), clone_op );
    }
    return matcher;
}

void TMMatcher::knnMatchImpl( const Mat& queryDescriptors, vector<vector<DMatch> >& matches, int k,
           const vector<Mat>& masks, bool compactResult )
{

#ifndef _DEMO_
	static Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ClassificationEngine.FeatureMatchTemplate"));
#endif

	//
	// Return all matches by thresholds.
	//

	std::vector< DMatch > good_matches;
	std::vector< DMatch > matches_2;
	int max_dist = 0; int min_dist = 20000;
	std::vector<KeyPoint> good_keypoints;

	int angle_dist, mean_dist;
	int descriptor_size = queryDescriptors.step;
	int query_desc_vector_size = *queryDescriptors.size.p;

	size_t s1=m_trainKeypoints->size(), s2=*trainDescCollection[0].size.p;

	CV_Assert( queryDescriptors.type() == trainDescCollection[0].type() );
	CV_Assert( m_trainKeypoints->size() == *trainDescCollection[0].size.p );

	for (int ind=0; ind<query_desc_vector_size; ind++)
	{
		good_matches.clear();
		good_keypoints.clear();
		matches_2.clear();
		max_dist = 0; min_dist = 20000;

		for (size_t i = 0; i < m_trainKeypoints->size(); i++)
		{
			uchar* size_i = trainDescCollection[0].ptr((int)i);
			const uchar* size_t = queryDescriptors.ptr((int)ind);
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
			mean_dist = std::abs(size_i[descriptor_size-1]-size_t[descriptor_size-1]);

			//-- Quick calculation of max and min distances between keypoints
			if( angle_dist < min_dist ) min_dist = angle_dist;
			if( angle_dist > max_dist ) max_dist = angle_dist;

			if( angle_dist > DESCRIPTOR_THRESHOLD_ANGLE && mean_dist < DESCRIPTOR_THRESHOLD_MEAN )
			{ 
				if (good_keypoints.empty())
				{
					good_matches.push_back(matches_2[i]);
					good_keypoints.push_back( m_trainKeypoints->at(matches_2[i].queryIdx) );
				}
				else
				{
					bool _insert = false;
					for (int k=0; k<good_keypoints.size(); k++)
					{
						float _norm = std::sqrt(std::pow(good_keypoints.at(k).pt.x - m_trainKeypoints->at(i).pt.x,2) +
												std::pow(good_keypoints.at(k).pt.y - m_trainKeypoints->at(i).pt.y,2));

						if (_norm > MIN_DIST_BETWEEN_KEYPOINTS)
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
								good_keypoints.push_back( m_trainKeypoints->at(matches_2[i].queryIdx) );

							}
							_insert = false;
							break;
						}
					}
					if(_insert)
					{
						good_matches.push_back( matches_2[i]);
						good_keypoints.push_back( m_trainKeypoints->at(matches_2[i].queryIdx) );
					}
				}
			}
		}

		matches.push_back(good_matches);

		LOG4CPLUS_INFO(logger, "-- Max dist : " << max_dist);
		LOG4CPLUS_INFO(logger, "-- Min dist : " << min_dist);
	}
}

void TMMatcher::knnMatchImpl(InputArray _queryDescriptors, std::vector<std::vector<DMatch> >& matches, int k,
	InputArrayOfArrays masks, bool compactResult)
{

#ifndef _DEMO_
	static Logger logger = Logger::getInstance(LOG4CPLUS_TEXT("ClassificationEngine.FeatureMatchTemplate"));
#endif

	//
	// Return all matches by thresholds.
	//

	std::vector< DMatch > good_matches;
	std::vector< DMatch > matches_2;
	int max_dist = 0; int min_dist = 20000;
	std::vector<KeyPoint> good_keypoints;

	Mat queryDescriptors(_queryDescriptors.getMat());

	int angle_dist, mean_dist;
	int descriptor_size = queryDescriptors.step;
	int query_desc_vector_size = *queryDescriptors.size.p;

	size_t s1 = m_trainKeypoints->size(), s2 = *trainDescCollection[0].size.p;

	CV_Assert(queryDescriptors.type() == trainDescCollection[0].type());
	CV_Assert(m_trainKeypoints->size() == *trainDescCollection[0].size.p);

	for (int ind = 0; ind < query_desc_vector_size; ind++)
	{
		good_matches.clear();
		good_keypoints.clear();
		matches_2.clear();
		max_dist = 0; min_dist = 20000;

		for (size_t i = 0; i < m_trainKeypoints->size(); i++)
		{
			uchar* size_i = trainDescCollection[0].ptr((int)i);
			const uchar* size_t = queryDescriptors.ptr((int)ind);
			angle_dist = 0;
			for (int j = 0; j < descriptor_size - 1; j++)
			{
				uchar xor_bit_diff = (size_i[j] ^ size_t[j]);
				uchar bit_diff = ~(size_i[j] ^ size_t[j]);//size_i[j] & size_t[j];
				if (bit_diff != 0)
				{
					for (int k = 0; k < 8; k++)
						angle_dist += (bit_diff & 1 << k) != 0;
				}
			}

			matches_2.push_back(DMatch(i, 0, (float)angle_dist));
			mean_dist = std::abs(size_i[descriptor_size - 1] - size_t[descriptor_size - 1]);

			//-- Quick calculation of max and min distances between keypoints
			if (angle_dist < min_dist) min_dist = angle_dist;
			if (angle_dist > max_dist) max_dist = angle_dist;

			if (angle_dist > DESCRIPTOR_THRESHOLD_ANGLE && mean_dist < DESCRIPTOR_THRESHOLD_MEAN)
			{
				if (good_keypoints.empty())
				{
					good_matches.push_back(matches_2[i]);
					good_keypoints.push_back(m_trainKeypoints->at(matches_2[i].queryIdx));
				}
				else
				{
					bool _insert = false;
					for (int k = 0; k < good_keypoints.size(); k++)
					{
						float _norm = std::sqrt(std::pow(good_keypoints.at(k).pt.x - m_trainKeypoints->at(i).pt.x, 2) +
							std::pow(good_keypoints.at(k).pt.y - m_trainKeypoints->at(i).pt.y, 2));

						if (_norm > MIN_DIST_BETWEEN_KEYPOINTS)
						{
							_insert = true;
						}
						else
						{
							if (matches_2[i].distance > good_matches.at(k).distance)
							{
								good_matches.erase(good_matches.begin() + k);
								good_keypoints.erase(good_keypoints.begin() + k);

								good_matches.push_back(matches_2[i]);
								good_keypoints.push_back(m_trainKeypoints->at(matches_2[i].queryIdx));

							}
							_insert = false;
							break;
						}
					}
					if (_insert)
					{
						good_matches.push_back(matches_2[i]);
						good_keypoints.push_back(m_trainKeypoints->at(matches_2[i].queryIdx));
					}
				}
			}
		}

		matches.push_back(good_matches);

		LOG4CPLUS_INFO(logger, "-- Max dist : " << max_dist);
		LOG4CPLUS_INFO(logger, "-- Min dist : " << min_dist);
	}
}

void TMMatcher::radiusMatchImpl( const Mat& queryDescriptors, vector<vector<DMatch> >& matches, float maxDistance,
           const vector<Mat>& masks, bool compactResult )
{
	return;
}

void TMMatcher::radiusMatchImpl(InputArray queryDescriptors, std::vector<std::vector<DMatch> >& matches, float maxDistance,
	InputArrayOfArrays masks, bool compactResult)
{
	return;
}
