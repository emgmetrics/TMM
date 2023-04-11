#ifndef _TMM_MATCHER_H_
#define _TMM_MATCHER_H_

#include <stdio.h>
#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;

typedef std::vector< std::pair<cv::Point, float> > MatchesDescriptionVector;

extern float thresholdMomentsDescriptors (std::vector<cv::DMatch> &dvector, double threshold);
extern MatchesDescriptionVector TMMFeatureMatchTemplate(cv::Mat& img_1,	cv::Mat& img_2,
									cv::Mat &mask,
									double threshold,
									int max_num_of_matches_to_find);

extern bool _IMG_SHOW_;
extern std::string save_files_dir_name;

namespace cv
{

    //
    // TMM descriptor Matcher.
    // 
    // For each descriptor in the first set, this matcher finds the closest
    // descriptor in the second set by trying each one.
    // 
    // 
    class CV_EXPORTS_W TMMatcher : public DescriptorMatcher
    {
    public:
        CV_WRAP TMMatcher( std::vector<KeyPoint>* trainKeypoints, bool crossCheck=false );
        virtual ~TMMatcher() {}

        virtual bool isMaskSupported() const { return false; }

        virtual Ptr<DescriptorMatcher> clone( bool emptyTrainData=false ) const;
   

    protected:
        virtual void knnMatchImpl( const Mat& queryDescriptors, vector<vector<DMatch> >& matches, int k,
               const vector<Mat>& masks=vector<Mat>(), bool compactResult=false );
    
        virtual void radiusMatchImpl( const Mat& queryDescriptors, vector<vector<DMatch> >& matches, float maxDistance,
               const vector<Mat>& masks=vector<Mat>(), bool compactResult=false );

        virtual void knnMatchImpl(InputArray queryDescriptors, std::vector<std::vector<DMatch> >& matches, int k,
            InputArrayOfArrays masks = noArray(), bool compactResult = false);
    
        virtual void radiusMatchImpl(InputArray queryDescriptors, std::vector<std::vector<DMatch> >& matches, float maxDistance,
            InputArrayOfArrays masks = noArray(), bool compactResult = false);

        bool m_crossCheck;
	    std::vector<KeyPoint> *m_trainKeypoints;
    };

} // end namespace cv
#endif