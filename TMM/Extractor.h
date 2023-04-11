#ifndef _TMM_EXTRACTOR_H_
#define _TMM_EXTRACTOR_H_

#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

#define _USE_SYS_CONFIG_FILE_
#if _DEMO_
#undef _USE_SYS_CONFIG_FILE_
#endif

#define MIN_TEMPLATE_WIDTH 200
#define MIN_TEMPLATE_HIGHT 70
#define _FEATURE_MATCH_SAVE_IMAGES true
#define _TMM_DESCRIPTOR_SIZE 64
#define _MOMENTS_DESCRIPTOR_SIZE 64
#define _FAST_THRESHOLD 55
#define __DESCRIPTOR_THRESHOLD_ANGLE_ 15
#define _DESCRIPTOR_THRESHOLD_ANGLE ((_TMM_DESCRIPTOR_SIZE-1)*8 - __DESCRIPTOR_THRESHOLD_ANGLE_)
#define _DESCRIPTOR_THRESHOLD_MEAN 60
#define _MIN_DIST_BETWEEN_KEYPOINTS 10.0f
#define _DEFAULT_NUM_OF_MATCHES_TO_FIND 20
#define _NUM_OF_TEMPLATE_KEYPOINTS_DEVIATION 10
#define _FILTER_ALGORITHM "Pixel-Pixel"
#define _FILTER_THRESHOLD_TOLERANCE 0.001
#define _DEFAULT_GLOBAL_USER_THRESHOLD 1.00

#ifdef _USE_SYS_CONFIG_FILE_
#include "SystemConfiguration.h"
#endif

class FeatureTemplateMatchConfiguration
{
public:
	FeatureTemplateMatchConfiguration()
	{
#ifdef _USE_SYS_CONFIG_FILE_
		try
		{
			SystemConfiguration& _config = SystemConfiguration::Instance();
			m_TMMDescriptorSize = SystemConfiguration::Instance().GetIntParam("TMMEngine.FeatureTemplateMatcher.TMMDescriptorSize");
			m_MomentsDescriptorSize = SystemConfiguration::Instance().GetIntParam("TMMEngine.FeatureTemplateMatcher.MomentsDescriptorSize");
			m_FASTThreshold = SystemConfiguration::Instance().GetIntParam("TMMEngine.FeatureTemplateMatcher.FASTThreshold");
			m_DescriptorThreshold_Angle = SystemConfiguration::Instance().GetIntParam("TMMEngine.FeatureTemplateMatcher.DescriptorThreshold_Angle");
			m_DescriptorThreshold_Mean = SystemConfiguration::Instance().GetIntParam("TMMEngine.FeatureTemplateMatcher.DescriptorThreshold_Mean");
			m_MinDistanceBetweenKeypoints = SystemConfiguration::Instance().GetFloatParam("TMMEngine.FeatureTemplateMatcher.MinDistanceBetweenKeypoints");

			m_DefaultNumOfMatchesToFind = SystemConfiguration::Instance().GetIntParam("TMMEngine.FeatureTemplateMatcher.DefaultNumOfMatchesToFind");

			m_SaveImages = SystemConfiguration::Instance().GetBoolParam("TMMEngine.FeatureTemplateMatcher.SaveImages");
			m_UseAllwaysOnFind = SystemConfiguration::Instance().GetBoolParam("TMMEngine.FeatureTemplateMatcher.UseAllwaysOnFind");

			m_NumOfTemplateKeypointsDeviation = _NUM_OF_TEMPLATE_KEYPOINTS_DEVIATION;

			m_FilterAlgorithm = SystemConfiguration::Instance().GetStringParam("TMMEngine.FeatureTemplateMatcher.FilterAlgorithm");
			m_FilterThresholdTolerance = SystemConfiguration::Instance().GetFloatParam("TMMEngine.FeatureTemplateMatcher.FilterThresholdTolerance");

			m_DefaultGlobalUserThreshold = SystemConfiguration::Instance().GetFloatParam("TMMEngine.FeatureTemplateMatcher.DefaultGlobalUserThreshold");	
		}
		catch (...)
		{
#endif
			m_TMMDescriptorSize = _TMM_DESCRIPTOR_SIZE;
			m_MomentsDescriptorSize = _MOMENTS_DESCRIPTOR_SIZE;
			m_FASTThreshold = _FAST_THRESHOLD;
			m_DescriptorThreshold_Angle = __DESCRIPTOR_THRESHOLD_ANGLE_;
			m_DescriptorThreshold_Mean = _DESCRIPTOR_THRESHOLD_MEAN;
			m_MinDistanceBetweenKeypoints = _MIN_DIST_BETWEEN_KEYPOINTS;
			m_DefaultNumOfMatchesToFind = _DEFAULT_NUM_OF_MATCHES_TO_FIND;
			m_SaveImages = false;
			m_UseAllwaysOnFind = false;
			m_NumOfTemplateKeypointsDeviation = _NUM_OF_TEMPLATE_KEYPOINTS_DEVIATION;
			m_FilterAlgorithm = _FILTER_ALGORITHM;
			m_FilterThresholdTolerance = _FILTER_THRESHOLD_TOLERANCE;
			m_DefaultGlobalUserThreshold = _DEFAULT_GLOBAL_USER_THRESHOLD;
#ifdef _USE_SYS_CONFIG_FILE_
		};
#endif
	}

	static FeatureTemplateMatchConfiguration& Instance()
	{
		static FeatureTemplateMatchConfiguration _FeatureTemplateMatchConfiguration;
		return _FeatureTemplateMatchConfiguration;
	}

	static void SetThresholds (double threshold);

	static int   m_TMMDescriptorSize;
	static int   m_MomentsDescriptorSize;
	static int   m_FASTThreshold;
	static int   m_DescriptorThreshold_Angle;
	static int   m_DescriptorThreshold_Mean;
	static float m_MinDistanceBetweenKeypoints;
	static int   m_DefaultNumOfMatchesToFind;
	static bool  m_SaveImages;
	static bool  m_UseAllwaysOnFind;
	static int   m_NumOfTemplateKeypointsDeviation;
	static std::string m_FilterAlgorithm;
	static double m_FilterThresholdTolerance;
	static float m_DefaultGlobalUserThreshold;
};

#define TMM_DESCRIPTOR_SIZE FeatureTemplateMatchConfiguration::m_TMMDescriptorSize
#define MOMENTS_DESCRIPTOR_SIZE FeatureTemplateMatchConfiguration::m_MomentsDescriptorSize
#define FAST_THRESHOLD FeatureTemplateMatchConfiguration::m_FASTThreshold
#define _DESCRIPTOR_THRESHOLD_ANGLE_ FeatureTemplateMatchConfiguration::m_DescriptorThreshold_Angle
#define DESCRIPTOR_THRESHOLD_ANGLE ((TMM_DESCRIPTOR_SIZE-1)*8 - _DESCRIPTOR_THRESHOLD_ANGLE_)
#define DESCRIPTOR_THRESHOLD_MEAN FeatureTemplateMatchConfiguration::m_DescriptorThreshold_Mean
#define MIN_DIST_BETWEEN_KEYPOINTS FeatureTemplateMatchConfiguration::m_MinDistanceBetweenKeypoints
#define FEATURE_MATCH_SAVE_IMAGES FeatureTemplateMatchConfiguration::m_SaveImages
#define DEFAULT_NUM_OF_MATCHES_TO_FIND FeatureTemplateMatchConfiguration::m_DefaultNumOfMatchesToFind
#define NUM_OF_TEMPLATE_KEYPOINTS_DEVIATION FeatureTemplateMatchConfiguration::m_NumOfTemplateKeypointsDeviation
#define FILTER_ALGORITHM FeatureTemplateMatchConfiguration::m_FilterAlgorithm
#define FILTER_THRESHOLD_TOLERANCE FeatureTemplateMatchConfiguration::m_FilterThresholdTolerance
#define DEFAULT_GLOBAL_USER_THRESHOLD FeatureTemplateMatchConfiguration::m_DefaultGlobalUserThreshold

using namespace std;

enum DetectorType
{
	DETECTOR_FAST,
	DETECTOR_HARRIS,
	DETECTOR_GFTT_WITH_HARRIS
};

class Mask
{
public:
	Mask(bool _use_feature_template_match=false, cv::Mat _mask=cv::Mat()):
	  m_UseFeatureTemplateMatch(_use_feature_template_match),
	  m_Mask(_mask)
	  {}

	bool m_UseFeatureTemplateMatch;
	cv::Mat& m_Mask;
};

namespace cv
{

	class CV_EXPORTS_W TMMDescriptorExtractor : public DescriptorExtractor
	{
	public:

		// the size of the signature in bytes
		///enum { kBytes = FeatureTemplateMatchConfiguration::m_TMMDescriptorSize };//TMM_DESCRIPTOR_SIZE };

		// constructors
		CV_WRAP explicit TMMDescriptorExtractor(Rect _template_rect = Rect(0, 0, 0, 0),
			Point2f _dist = Point2f(0.f, 0.f), Mat _mask = Mat());


		// returns the descriptor size in bytes
		int descriptorSize() const;

		// returns the descriptor type
		int descriptorType() const;

		// Compute the TMM features and descriptors on an image
		void operator()(InputArray image, InputArray mask, vector<KeyPoint>& keypoints) const;

		// Compute the TMM features and descriptors on an image
		void operator()( InputArray image, InputArray mask, vector<KeyPoint>& keypoints,
						 OutputArray descriptors, bool useProvidedKeypoints=false ) const;


		// Compute descriptor for a template image 
		void computeTemplateDescriptor (const Mat& template_image, vector<KeyPoint>& keypoints,
										 Mat& descriptor, const Mat& mask);


		// Compute moments descriptor for a template image 
		void computeTemplateMomentsDescriptor (const Mat& template_image, vector<KeyPoint>& keypoints,	 Mat& descriptor);

		// Compute TMM descriptor moments for a set of keypoints
		void computeKeypointsMomentsDescriptors (const Mat& image, vector<KeyPoint>& keypoints,
										 Mat& descriptor, const Mat& mask);

		// return the distance of the base template keypoint to the (0,0).
		Point getDist();

		// set the type of the detector engine to use for calculating feature points.
		void setCurrentDetectorType (DetectorType detectorType);

		// get the currenttype of the detector engine that is used for calculating feature points.
		DetectorType getCurrentDetectorType ();

		// return the maximum/minimun number of keypoints deviation permited of a template patch from the current Template.
		int GetMaxTemplateKepointsTolerance();
		int GetMinTemplateKepointsTolerance();

		CV_WRAP virtual void compute(InputArray image,
			CV_OUT CV_IN_OUT std::vector<KeyPoint>& keypoints,
			OutputArray descriptors);
 
		void computeImpl( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) const;

	protected:


		CV_PROP_RW cv::Rect m_Template_rect;
		CV_PROP_RW Point2f m_Dist;
		CV_PROP_RW int m_Number_of_templ_keypoints;

		CV_PROP_RW Mat m_Mask; // Matrix with non-zero values in the region of interest (0=masked, 1=no-mask).
		CV_PROP_RW DetectorType m_DetectorType;


	};

} // end namespace cv

#endif