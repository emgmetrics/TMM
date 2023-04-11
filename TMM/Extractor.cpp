#include "Extractor.h"

#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "Timer.h"

using namespace std;
using namespace cv;

//#define _MOMENTS_BY_KEYPOINTS_
#define HARRIS_MAX_N_FEATURES 50000


int    FeatureTemplateMatchConfiguration::m_TMMDescriptorSize = _TMM_DESCRIPTOR_SIZE;
int    FeatureTemplateMatchConfiguration::m_MomentsDescriptorSize = _MOMENTS_DESCRIPTOR_SIZE;
int    FeatureTemplateMatchConfiguration::m_FASTThreshold = _FAST_THRESHOLD;
int    FeatureTemplateMatchConfiguration::m_DescriptorThreshold_Angle = __DESCRIPTOR_THRESHOLD_ANGLE_;
int    FeatureTemplateMatchConfiguration::m_DescriptorThreshold_Mean = _DESCRIPTOR_THRESHOLD_MEAN;
float  FeatureTemplateMatchConfiguration::m_MinDistanceBetweenKeypoints = _MIN_DIST_BETWEEN_KEYPOINTS;
int    FeatureTemplateMatchConfiguration::m_DefaultNumOfMatchesToFind = _DEFAULT_NUM_OF_MATCHES_TO_FIND;
bool   FeatureTemplateMatchConfiguration::m_SaveImages = _FEATURE_MATCH_SAVE_IMAGES;
bool   FeatureTemplateMatchConfiguration::m_UseAllwaysOnFind = false;
int    FeatureTemplateMatchConfiguration::m_NumOfTemplateKeypointsDeviation = _NUM_OF_TEMPLATE_KEYPOINTS_DEVIATION;
string FeatureTemplateMatchConfiguration::m_FilterAlgorithm = _FILTER_ALGORITHM;
double FeatureTemplateMatchConfiguration::m_FilterThresholdTolerance = _FILTER_THRESHOLD_TOLERANCE;
float  FeatureTemplateMatchConfiguration::m_DefaultGlobalUserThreshold = _DEFAULT_GLOBAL_USER_THRESHOLD;



#define NUM_OF_THRESHOLS_RANGES 3
void FeatureTemplateMatchConfiguration::SetThresholds(double threshold)
{
	double threshold_ranges[NUM_OF_THRESHOLS_RANGES] = {0.3, 0.7, 1.0};
	int thrsh_range = 0;

	for (thrsh_range=0; thrsh_range<NUM_OF_THRESHOLS_RANGES; thrsh_range++)
		if (threshold < threshold_ranges[thrsh_range])
			break;
			
	switch (thrsh_range)
	{
		case 0: // 0<= threshold <  0.3
			NUM_OF_TEMPLATE_KEYPOINTS_DEVIATION = 9999;
			_DESCRIPTOR_THRESHOLD_ANGLE_ = 100;
			break;
		case 1: // 03 <= threshold <  0.7
			NUM_OF_TEMPLATE_KEYPOINTS_DEVIATION = 40;
			_DESCRIPTOR_THRESHOLD_ANGLE_ = 50;
			break;
		case 2: // 70 <== threshold <  1.0
			NUM_OF_TEMPLATE_KEYPOINTS_DEVIATION = 20;
			_DESCRIPTOR_THRESHOLD_ANGLE_ = 20;
			break;
		case 3: // threshold == 1.0
			NUM_OF_TEMPLATE_KEYPOINTS_DEVIATION = _NUM_OF_TEMPLATE_KEYPOINTS_DEVIATION;
			_DESCRIPTOR_THRESHOLD_ANGLE_ = __DESCRIPTOR_THRESHOLD_ANGLE_;
			break;
		default:
			// do nothing and live default values.
			break;
	}
}

Ptr<FeatureDetector> CreateFeatureDetector (DetectorType detectorType)
{
	Ptr<FeatureDetector> detector;

	switch(detectorType)
	{
		case DETECTOR_FAST:
			{
				detector = FastFeatureDetector::create(FAST_THRESHOLD);
				//FastFeatureDetector* fast_detector = FastFeatureDetector::create(FAST_THRESHOLD);
				//detector = fast_detector;
			}
			break;
		//case DETECTOR_HARRIS:
		//	detector = FeatureDetector::create("HARRIS");
		//	break;
		case DETECTOR_GFTT_WITH_HARRIS:
			{
				GFTTDetector* gftt_detector = GFTTDetector::create(HARRIS_MAX_N_FEATURES, 0.000001, 0, 3, true, 0.04);
				detector = gftt_detector;
			}
			break;
		default:
			detector = NULL;
	}

	return detector;
}

static bool compareByAngle(const KeyPoint keypoint1, const KeyPoint keypoint2)
{
	return ( keypoint1.angle > keypoint2.angle );
}

static bool compareByResponse(const KeyPoint keypoint1, const KeyPoint keypoint2)
{
	return ( keypoint1.response > keypoint2.response );
}

static bool compareByX(const KeyPoint keypoint1, const KeyPoint keypoint2)
{
	if (keypoint1.pt.x < keypoint2.pt.x)
		return true;
	return false;
}

static bool compareByXY(const KeyPoint keypoint1, const KeyPoint keypoint2)
{
	if (keypoint1.pt.x < keypoint2.pt.x)
		return true;
	else if ((keypoint1.pt.x == keypoint2.pt.x) && (keypoint1.pt.y < keypoint2.pt.y))
		return true;
	return false;
}

static bool compareByY(const KeyPoint keypoint1, const KeyPoint keypoint2)
{
	if (keypoint1.pt.y < keypoint2.pt.y)
		return true;
	return false;
}

static bool compareByCoordinates(const KeyPoint keypoint1, const KeyPoint keypoint2)
{
	if (keypoint1.pt.y > keypoint2.pt.y)
		return true;
	else if ((keypoint1.pt.y == keypoint2.pt.y) && (keypoint1.pt.x < keypoint2.pt.x))
		return true;
	return false;
}

static void sortKeypointsByAngle (std::vector<KeyPoint> *keypoints)
{
	std::sort(keypoints->begin(), keypoints->end(), compareByAngle);
}

static void sortKeypointsByResponse (std::vector<KeyPoint> *keypoints)
{
	std::sort(keypoints->begin(), keypoints->end(), compareByResponse);
}

static void sortKeypointsByCoordinates (std::vector<KeyPoint> *keypoints)
{
	std::sort(keypoints->begin(), keypoints->end(), compareByCoordinates);
}

static void sortKeypointsByX (std::vector<KeyPoint> *keypoints)
{
	std::sort(keypoints->begin(), keypoints->end(), compareByX);
}

static void sortKeypointsByY (std::vector<KeyPoint> *keypoints)
{
	std::sort(keypoints->begin(), keypoints->end(), compareByY);
}

static void sortKeypointsByXY (std::vector<KeyPoint> *keypoints)
{
	std::sort(keypoints->begin(), keypoints->end(), compareByXY);
}

static void _computeKeypointsAngle (std::vector<KeyPoint> *keypoints, float base_x, float base_y)
{
	static float TMMDescriptorSize = TMM_DESCRIPTOR_SIZE;
	static float angle_scale = ((float)((TMMDescriptorSize-1.0f)*8.0f))/90.0f;
	for (unsigned int i=0; i<keypoints->size(); i++)
		keypoints->at(i).angle = (float)cvFloor((float)(angle_scale*fastAtan2(
													abs(base_y - keypoints->at(i).pt.y),
													abs(base_x - keypoints->at(i).pt.x))));									
}

static void _computeTMMDescriptor(const Mat& image, vector<KeyPoint>&keypoints, uchar* desc, int desc_size, const Mat& mask)
{
	unsigned char angle, byte_pos, bit_pos;
	float image_mean = 0;

	int keypoints_vect_size = keypoints.size();
	for (int i=0; i<keypoints_vect_size; i++)
	{
		angle    = (uchar)keypoints[i].angle;
		byte_pos = angle / 8;
		bit_pos  = angle % 8;
		desc[byte_pos] |= 1 << bit_pos;
	}
	Scalar template_mean = mean(image, mask);
	desc[desc_size-1] = template_mean[0];
}


#define LOG_AND_COPYSGN(val) (val>=-1&&val<=1)?(val*1000):(_copysign(log10(abs(val)),val)+_copysign(1000,val));
//#define LOG_AND_COPYSGN(val) _copysign(val==0.0f?0:log10(abs(val)), val);
//#define LOG_AND_COPYSGN(val) atan(val);
//#define LOG_AND_COPYSGN(val) (val);
//#define EXP_H_PI 37.2217104
//#define LOG_AND_COPYSGN(val) ((val>=EXP_H_PI || val<=-1*EXP_H_PI)?_copysign(val==0.0f?0:log10(abs(val)), val):atan(val));

static void _computeTMMDescriptorMoments(const Mat& image, vector<KeyPoint>&keypoints, float* desc, const Mat& mask)
{

	Mat kpoints_mat (image.rows, image.cols, image.type());
	kpoints_mat = 0;
	
#ifdef _MOMENTS_BY_KEYPOINTS_
	// Calculate Moments only on keypoints.
	int keypoints_vect_size = keypoints.size();
	for (unsigned int i=0; i<keypoints_vect_size; i++)
	{
		kpoints_mat.at<uchar>(Point(keypoints[i].pt.x,keypoints[i].pt.y)) = image.at<uchar>(Point(keypoints[i].pt.x,keypoints[i].pt.y));
	}
#else
	if( mask.empty() )
	{
		for (int i=0; i<image.rows; i++)
			for (int j=0; j<image.cols; j++)
				kpoints_mat.at<uchar>(Point(j,i)) = image.at<uchar>(Point(j,i));
	}
	else
	{
		for (int i=0; i<image.rows; i++)
			for (int j=0; j<image.cols; j++)
			{
				if (mask.at<uchar>(Point(j,i)))
				{
					kpoints_mat.at<uchar>(Point(j,i)) = image.at<uchar>(Point(j,i));
				}
			}
	}
#endif

	Moments keypoints_moments = moments(kpoints_mat);

#if 0
	double hu_keypoints_moments[7];
	HuMoments(keypoints_moments, hu_keypoints_moments);

	desc[0] = _copysign(log10(abs(hu_keypoints_moments[0])), hu_keypoints_moments[0]);
	desc[1] = _copysign(log10(abs(hu_keypoints_moments[1])), hu_keypoints_moments[1]);
	desc[2] = _copysign(log10(abs(hu_keypoints_moments[2])), hu_keypoints_moments[2]);
	desc[3] = _copysign(log10(abs(hu_keypoints_moments[3])), hu_keypoints_moments[3]);
	desc[4] = _copysign(log10(abs(hu_keypoints_moments[4])), hu_keypoints_moments[4]);
	desc[5] = _copysign(log10(abs(hu_keypoints_moments[5])), hu_keypoints_moments[5]);
	desc[6] = _copysign(log10(abs(hu_keypoints_moments[6])), hu_keypoints_moments[6]);
#else
	desc[0] = LOG_AND_COPYSGN(keypoints_moments.m00);
	desc[1] = LOG_AND_COPYSGN(keypoints_moments.m01);
	desc[2] = LOG_AND_COPYSGN(keypoints_moments.m02);
	desc[3] = LOG_AND_COPYSGN(keypoints_moments.m03);
	desc[4] = LOG_AND_COPYSGN(keypoints_moments.m10);
	desc[5] = LOG_AND_COPYSGN(keypoints_moments.m11);
	desc[6] = LOG_AND_COPYSGN(keypoints_moments.m12);
	desc[7] = LOG_AND_COPYSGN(keypoints_moments.m20);
	desc[8] = LOG_AND_COPYSGN(keypoints_moments.m21);
	desc[9] = LOG_AND_COPYSGN(keypoints_moments.m30);

	desc[10] = LOG_AND_COPYSGN(keypoints_moments.mu02);
	desc[11] = LOG_AND_COPYSGN(keypoints_moments.mu03);
	desc[12] = LOG_AND_COPYSGN(keypoints_moments.mu11);
	desc[13] = LOG_AND_COPYSGN(keypoints_moments.mu12);
	desc[14] = LOG_AND_COPYSGN(keypoints_moments.mu20);
	desc[15] = LOG_AND_COPYSGN(keypoints_moments.mu21);
	desc[16] = LOG_AND_COPYSGN(keypoints_moments.mu30);

	desc[17] = LOG_AND_COPYSGN(keypoints_moments.nu02);
	desc[18] = LOG_AND_COPYSGN(keypoints_moments.nu03);
	desc[19] = LOG_AND_COPYSGN(keypoints_moments.nu11);
	desc[20] = LOG_AND_COPYSGN(keypoints_moments.nu12);
	desc[21] = LOG_AND_COPYSGN(keypoints_moments.nu20);
	desc[22] = LOG_AND_COPYSGN(keypoints_moments.nu21);
	desc[23] = LOG_AND_COPYSGN(keypoints_moments.nu30);
#endif
}

void _computeKeypointsMomentsDescriptors (
			const Mat& image, vector<KeyPoint>& keypoints,
			Mat& descriptors, 
			Rect template_rect,
			Point2f dist,
			const Mat& mask)
{

	CvRect keypoint_roi;
	int delta = 0;//px

	int descriptor_size = MOMENTS_DESCRIPTOR_SIZE;
	descriptors = Mat::zeros((int)keypoints.size(), descriptor_size, CV_32F);

    for (size_t i = 0; i < keypoints.size(); i++)
	{

		float x = keypoints[i].pt.x;
		float y = keypoints[i].pt.y;
		keypoint_roi.x = (int)(keypoints[i].pt.x - dist.x) - delta;
		keypoint_roi.y = (int)(keypoints[i].pt.y - dist.y) - delta;
		keypoint_roi.width  = template_rect.width + delta;
		keypoint_roi.height = template_rect.height + delta;

#if 0
		keypoint_roi.x = keypoint_roi.x<=0?0:(int)(keypoint_roi.x);
		keypoint_roi.y = keypoint_roi.y<=0?0:(int)(keypoint_roi.y);
		keypoint_roi.width  = keypoint_roi.x+template_rect.width+delta>=image.cols?image.cols-keypoint_roi.x:keypoint_roi.width;
		keypoint_roi.height = keypoint_roi.y+template_rect.height+delta>=image.rows?image.rows-keypoint_roi.y:keypoint_roi.height;

#else
		if (keypoint_roi.x<0 || keypoint_roi.x+keypoint_roi.width>=image.cols ||
			keypoint_roi.y<0 || keypoint_roi.y+keypoint_roi.height>=image.rows)
		{
			continue;
		}
#endif

		Mat tmp_mask = Mat();
		if (!mask.empty())
		{
			if (mask.cols!=keypoint_roi.width || mask.rows!=keypoint_roi.height)
			{
				Rect rect(0,0,keypoint_roi.width, keypoint_roi.height);
				tmp_mask = mask(rect);
			}
			else
				tmp_mask = mask;
		}


		Mat roi_mat = image(keypoint_roi);

		_computeTMMDescriptorMoments(roi_mat, keypoints, (float*)descriptors.ptr((int)i), tmp_mask);

	}
}

static void _computeTemplateDescriptor(
		const Mat& template_image,
		vector<KeyPoint>& keypoints,
		DetectorType detectorType,
		void* descriptor,
		int descriptor_size,
		const Mat& mask,
		int max_templ_keypoints,
		int min_templ_keypoints,
		bool compute_moments=false,
		bool use_current_kepoints=false)
{


    //convert to grayscale if more than one color
    CV_Assert(template_image.type() == CV_8UC1);

	if (!use_current_kepoints)
	{
#ifdef _USE_ONLY_FAST_DETECTOR_
		FastFeatureDetector fast_detector(FAST_THRESHOLD);
		fast_detector.detect( template_image, keypoints, mask );
#else
		//cv::FAST(template_image, keypoints, FAST_THRESHOLD);
		Ptr<FeatureDetector> feature_detector = CreateFeatureDetector(detectorType);
		feature_detector->detect( template_image, keypoints, mask );
		feature_detector.release();
#endif
	}

	if (!keypoints.empty())
	{
		int kepoints_size = keypoints.size();
		if (kepoints_size<=max_templ_keypoints && kepoints_size>=min_templ_keypoints)
		{
			if (!compute_moments)
			{
				//sortKeypointsByCoordinates(&keypoints);
				_computeKeypointsAngle(&keypoints, 0,0/*keypoints[0].pt.x, keypoints[0].pt.y*/);
				//sortKeypointsByAngle(&keypoints);

				_computeTMMDescriptor(template_image, keypoints, (uchar*)descriptor, descriptor_size, mask);
			}
			else
			{
				_computeTMMDescriptorMoments(template_image, keypoints, (float*)descriptor, mask);
			}

		}
	}
}

static void _findKeypointsInsideRect(vector<KeyPoint>& keypoints, vector<KeyPoint>& keypoints_found, cv::Rect roi, size_t *Last_foundKeypointInsideRect)
{

	static KeyPoint _key_point(0, 0, 7.f, -1, 0.0f);
	size_t _vector_kepoints_size = keypoints.size();

	//
	// NOTE: vector keypoints must be sorted by X coordinate.
	//

	if (!Last_foundKeypointInsideRect)
		return;

	for (size_t i = (*Last_foundKeypointInsideRect); i < _vector_kepoints_size; i++)
	{
		if ( (keypoints[i].pt.x>roi.x+2) )
		{
			(*Last_foundKeypointInsideRect) = i;
			while ( i<_vector_kepoints_size && (keypoints[i].pt.x<roi.x+roi.width-2) )
			{
				if ( (keypoints[i].pt.y>roi.y+2) && (keypoints[i].pt.y<roi.y+roi.height-2) )
				{
					_key_point.pt.x = keypoints[i].pt.x - roi.x;
					_key_point.pt.y = keypoints[i].pt.y - roi.y;
					keypoints_found.push_back(_key_point);
				}
				i++;
			}
			break;
		}
	}
}

static void _computeDescriptors(const Mat& image, vector<KeyPoint>& keypoints,
								DetectorType detectorType,
								Mat& descriptors, int descriptor_size,
                                Rect template_rect, Point2f dist, const Mat& mask,
								int max_templ_keypoints, int min_templ_keypoints,
								bool compute_moments=false)
{
	Timer tmr;

    //convert to grayscale if more than one color
    CV_Assert(image.type() == CV_8UC1);

	//create the descriptor mat, keypoints.size()
	if (!compute_moments)
		descriptors = Mat::zeros((int)keypoints.size(), descriptor_size, CV_8U);
	else
	{
		descriptor_size = MOMENTS_DESCRIPTOR_SIZE;
		descriptors = Mat::zeros((int)keypoints.size(), descriptor_size, CV_32F);
	}

	// need to sort keypoints by X coordinates in order to perform faster
	// in function _findKeypointsInsideRect().
	sortKeypointsByXY(&keypoints);


	std::vector<KeyPoint> keypoints_found;
	keypoints_found.reserve(keypoints.size()/4);
	size_t Last_foundKeypointInsideRect = 0;

	cv::Rect keypoint_roi;
	int delta = 0;//px

    for (size_t i = 0; i < keypoints.size(); i++)
	{

		std::vector<KeyPoint> keypoints_roi;

		float x = keypoints[i].pt.x;
		float y = keypoints[i].pt.y;
		keypoint_roi.x = (int)(keypoints[i].pt.x - dist.x) - delta;
		keypoint_roi.y = (int)(keypoints[i].pt.y - dist.y) - delta;
		keypoint_roi.width  = template_rect.width + delta;
		keypoint_roi.height = template_rect.height + delta;

#if 0
		keypoint_roi.x = keypoint_roi.x<=0?0:(int)(keypoint_roi.x);
		keypoint_roi.y = keypoint_roi.y<=0?0:(int)(keypoint_roi.y);
		keypoint_roi.width  = keypoint_roi.x+template_rect.width+delta>=image.cols?image.cols-keypoint_roi.x:keypoint_roi.width;
		keypoint_roi.height = keypoint_roi.y+template_rect.height+delta>=image.rows?image.rows-keypoint_roi.y:keypoint_roi.height;

#else
		if (keypoint_roi.x<0 || keypoint_roi.x+keypoint_roi.width>=image.cols ||
			keypoint_roi.y<0 || keypoint_roi.y+keypoint_roi.height>=image.rows)
		{
			continue;
		}
#endif


		Mat tmp_mask = Mat();
		if (!mask.empty())
		{
			if (mask.cols!=keypoint_roi.width || mask.rows!=keypoint_roi.height)
			{
				cv::Rect rect(0,0,keypoint_roi.width, keypoint_roi.height);
				tmp_mask = mask(rect);
			}
			else
				tmp_mask = mask;
		}
#if 1
		keypoints_found.clear();

#ifdef _DEBUG
		tmr.Start();
		_findKeypointsInsideRect(keypoints, keypoints_found, keypoint_roi, &Last_foundKeypointInsideRect);
		tmr.Stop();
#else
		_findKeypointsInsideRect(keypoints, keypoints_found, keypoint_roi, &Last_foundKeypointInsideRect);
#endif

		KeyPointsFilter::runByPixelsMask( keypoints_found, tmp_mask );


		cv::Mat roi_mat = image(keypoint_roi);

		if (!compute_moments)
			_computeTemplateDescriptor(
				roi_mat,
				keypoints_found,
				detectorType,
				descriptors.ptr((int)i),
				descriptor_size,
				tmp_mask,
				max_templ_keypoints,
				min_templ_keypoints,
				compute_moments,
				true);
		else
			_computeTemplateDescriptor(
				roi_mat,
				keypoints_roi,
				detectorType,
				descriptors.ptr((int)i),
				descriptor_size,
				tmp_mask,
				max_templ_keypoints,
				min_templ_keypoints,
				compute_moments,
				false);

#else

		Mat roi_mat = image(keypoint_roi);

		_computeTemplateDescriptor(
			roi_mat,
			keypoints_roi,
			detectorType,
			descriptors.ptr((int)i),
			descriptor_size,
			tmp_mask,
			max_templ_keypoints,
			min_templ_keypoints,
			compute_moments,
			false);
#endif
	}

#ifdef _DEBUG
	cout << "_findKeypointsInsideRect accumulate elapse time: " << tmr.GetTime() << " sec." << endl;
#endif
}

TMMDescriptorExtractor::TMMDescriptorExtractor(Rect _template_rect, Point2f _dist, Mat _mask):
		m_Template_rect(_template_rect),
		m_Dist(_dist),
		m_Mask(_mask)
{
	m_Number_of_templ_keypoints = 0;
	m_DetectorType = DETECTOR_FAST;

	FeatureTemplateMatchConfiguration::Instance();
}

int TMMDescriptorExtractor::descriptorSize() const
{
	// the size of the signature in bytes
    return TMM_DESCRIPTOR_SIZE;
}

int TMMDescriptorExtractor::descriptorType() const
{
    return CV_8U;
}

int TMMDescriptorExtractor::GetMinTemplateKepointsTolerance ()
{
	return m_Number_of_templ_keypoints - m_Number_of_templ_keypoints/10 - FeatureTemplateMatchConfiguration::m_NumOfTemplateKeypointsDeviation;
}

int TMMDescriptorExtractor::GetMaxTemplateKepointsTolerance ()
{
	return m_Number_of_templ_keypoints + m_Number_of_templ_keypoints/10 + FeatureTemplateMatchConfiguration::m_NumOfTemplateKeypointsDeviation;
}

void TMMDescriptorExtractor::compute(InputArray image,
	CV_OUT CV_IN_OUT std::vector<KeyPoint>& keypoints,
	OutputArray descriptors)
{
	//computeImpl(image, keypoints, descriptors);
}

void TMMDescriptorExtractor::computeImpl( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) const
{

	_computeDescriptors(image, keypoints, 
						m_DetectorType,
						descriptors, descriptorSize(),
                        m_Template_rect, 
						m_Dist,
						m_Mask,
						((TMMDescriptorExtractor*)this)->GetMaxTemplateKepointsTolerance(),
						((TMMDescriptorExtractor*)this)->GetMinTemplateKepointsTolerance());
}

void TMMDescriptorExtractor::computeTemplateDescriptor (const Mat& template_image, vector<KeyPoint>& keypoints,
									 Mat& descriptor, const Mat& mask)
{

	//
	//create the template descriptor of size 1*descriptor_size*8
	//
    descriptor = Mat::zeros(1, descriptorSize(), CV_8U);

	//
	// compute template descriptor.
	//
	_computeTemplateDescriptor(template_image, keypoints, m_DetectorType, descriptor.ptr((int)0), descriptorSize(), mask, 9999, -9999);
	if (keypoints.empty())
	{
		descriptor.release();
		cout << "No keypoint found in Template image, return without computing Template Descriptor" << endl;
		return;
	}


	//
	// Initialize class members.
	//

	m_Number_of_templ_keypoints = keypoints.size();


	//
	// re-calculate new mask according to keypoints limits.
	//
	int min_x, max_x, min_y, max_y;
	sortKeypointsByX(&keypoints);
	min_x = keypoints.front().pt.x-2;
	max_x = keypoints.back().pt.x+2;

	sortKeypointsByY(&keypoints);
	min_y = keypoints.front().pt.y-2;
	max_y = keypoints.back().pt.y+2;

	if (mask.empty())
		m_Mask = Mat();
	else
	{
		m_Mask = mask.clone();
#if 0
		for (int i=0; i<m_Mask.cols; i++)
			for (int j=0; j<m_Mask.rows; j++)
				if ( (i<min_x || i>max_x) || (j<min_y || j>max_y) )
					m_Mask.at<uchar>(i + j*m_Mask.cols) = 0;
#endif
	}

	//
	// re-calculate template descriptor mean according to the new mask.
	//
	uchar* desc = descriptor.ptr((int)0);
	Scalar template_mean = mean(template_image, m_Mask);
	desc[descriptorSize()-1] = template_mean[0];

	m_Template_rect.x = 0;
	m_Template_rect.y = 0;
	m_Template_rect.height = template_image.rows;
	m_Template_rect.width  = template_image.cols;


	sortKeypointsByResponse(&keypoints);
	m_Dist.x = keypoints[0].pt.x;
	m_Dist.y = keypoints[0].pt.y;
}

void TMMDescriptorExtractor::computeTemplateMomentsDescriptor (const Mat& template_image, vector<KeyPoint>& keypoints,	 Mat& descriptor)
{

	//create the template descriptor
    descriptor = Mat::zeros(1, MOMENTS_DESCRIPTOR_SIZE, CV_32F);

	_computeTMMDescriptorMoments(template_image, keypoints, (float*)descriptor.ptr((int)0), m_Mask);

}

void TMMDescriptorExtractor::computeKeypointsMomentsDescriptors (const Mat& image, vector<KeyPoint>& keypoints,
									 Mat& descriptors, const Mat& mask)
{

    CV_Assert(image.type() == CV_8UC1);

#ifdef _MOMENTS_BY_KEYPOINTS_
	//
	// Calculate Moments descriptors by using only image Keypoints as computing points.
	//
	_computeDescriptors(image, keypoints,
						m_DetectorType,
						descriptors, descriptorSize(),
                        m_Template_rect, 
						m_Dist,
						m_Mask,
						GetMaxNumOfTemplateKepoints(),
						GetMinNumOfTemplateKepoints(),
						true);
#else
	//
	// Calculate Moments descriptors by using only image pixels intensity as computing points.
	//
	_computeKeypointsMomentsDescriptors(
						image, keypoints, 
						descriptors,
                        m_Template_rect, 
						m_Dist,
						m_Mask);
#endif

}

Point TMMDescriptorExtractor::getDist()
{
	return m_Dist;
}

void TMMDescriptorExtractor::setCurrentDetectorType (DetectorType detectorType)
{
	m_DetectorType = detectorType;
}

DetectorType TMMDescriptorExtractor::getCurrentDetectorType ()
{
	return m_DetectorType;
}