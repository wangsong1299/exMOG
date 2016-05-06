/************************************************************************
* Copyright(c) 2011  Yang Xian
* All rights reserved.
*
* File:	MOGExtractForeground.cpp
* Brief: 基于混合高斯模型的前景检测
* Version: 1.0
* Author: Yang Xian
* Email: xyang2011@sinano.ac.cn
* Date:	2011/11/19
* History:
************************************************************************/
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include "cv.h"
#include "highgui.h"
#include <time.h>
#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <iostream>

//#include "m_background_segm.hpp"

using namespace cv;
using namespace std;
const int CONTOUR_MAX_AERA = 100;
int lastAvg[5]={0};
int updateFrames=0;

int getUpdateFlag(int count,Mat frame){
	if(count<20){
		return 1;
	}
	if(updateFrames>0){
		updateFrames-=1;
		return 1;
	}
	int avg=0,sum=0;
	for (int j=0; j<frame.rows; j++) {
      uchar* data= frame.ptr<uchar>(j);
	  for (int i=0; i<frame.cols; i++) {
                  sum+=data[i];
      }                  
    }
	avg=sum/(240*640);
	for(int i=0;i<4;i++){
		lastAvg[i]=lastAvg[i+1];
	}
	lastAvg[4]=avg;
	if((lastAvg[0]!=0)&&(abs(lastAvg[1]-lastAvg[0])>10)&&(abs(lastAvg[2]-lastAvg[0])>10)&&(abs(lastAvg[3]-lastAvg[0])>10)&&(abs(lastAvg[4]-lastAvg[0])>10)){
		updateFrames=20;
		return 1;
	}
	return 0;
}

Mat gray_prev;
int main()
{
	Mat frame; 
	Mat foreground,background,gray,gray2,result2,foreground0;	// 前景图片
	Mat mask(240, 320, CV_8UC1,255);
	Mat liantong(240, 320, CV_8UC1,255);
	IplImage front,frameO,frame1;
	CvMemStorage *stor;
    CvSeq *cont, *result, *squares;
    CvSeqReader reader;
	int updateFlag=1;
	//part3
	Mat mask2(240, 320, CV_8UC1,255);

	VideoCapture capture("bike.avi");

	if (!capture.isOpened())
	{
		return 0;
	}

	namedWindow("part2");
	namedWindow("mask");
	namedWindow("Source Video");
	namedWindow("part1");
	namedWindow("background");
	namedWindow("temp");
	namedWindow("dst");
	namedWindow("result");
	namedWindow("mask2");
	namedWindow("part3");
	namedWindow("ws");
	//namedWindow("ws2");
	int count=0;
	// 混合高斯物体
	cv::BackgroundSubtractorMOG2 mog;
	bool stop(false);
	while (!stop)
	{
		if (!capture.read(frame))
		{
			break;
		}
		 
		if(count==0){
		// 更新背景图片并且输出前景
		mog(frame,mask,liantong,foreground,updateFlag,0.01);
		mog.getBackgroundImage(background);
		mask(Rect(1,1,318,238))=255;
		mask2(Rect(1,1,318,238))=255;
		}

		imshow("Source Video", frame);


		//part1
		//粗略提取运动区域
		cvtColor(frame,gray,CV_BGR2GRAY);
		cvtColor(background,gray2,CV_BGR2GRAY);
		absdiff(gray,gray2,result2);
		threshold(result2, result2, 10, 255, THRESH_BINARY_INV);
		imshow("part1", result2);

		//extra1
		//辅助操作，初始化和环境突变（非闪光灯）的时候整帧更新
		updateFlag=getUpdateFlag(count,frame);
		updateFlag=1;

		front=IplImage(result2);
		frameO=IplImage(frame);
		frame1=IplImage(frame);

		 //得到mask
		stor = cvCreateMemStorage(0);
		cont = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint) , stor);
		 //找到所有轮廓
		cvFindContours(&front, stor, &cont, sizeof(CvContour), 
						CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));
		// 直接使用CONTOUR中的矩形来画轮廓
		for(;cont;cont = cont->h_next)
		{
				  CvRect r = ((CvContour*)cont)->rect;
				  if((r.height * r.width > CONTOUR_MAX_AERA)&&(r.x!=1)) // 面积小的方形抛弃掉
				     {
					  mask(Rect(r.x,r.y,r.width,r.height))=0;
					  //cvRectangle(&frameO, cvPoint(r.x,r.y),cvPoint(r.x + r.width, r.y + r.height),CV_RGB(0,255,0), 1, CV_AA,0);
				  }
		}
		imshow("mask",mask);

		//part2 
		//精确提取运动区域
		// 更新背景图片并且输出前景
		mog(frame,mask,liantong,foreground,updateFlag,0.01);
		mog.getBackgroundImage(background);
		imshow("background", background);
		mask(Rect(1,1,318,238))=255;
		// 输出的前景图片并不是2值图片，要处理一下显示  
		threshold(foreground, foreground, 128, 255, THRESH_BINARY_INV);
		imshow("part2", foreground);

		//extra2
		//得到连通域特性mat
		int width=320,height=240;
		for (int j=0; j<height; j++){  
			int prevj=j-1,nextj=j+1;
			if(j==0){prevj=0;}
			if(j==height-1){nextj=height-1;}
			uchar* data= foreground.ptr<uchar>(j);
			uchar* dataprev= foreground.ptr<uchar>(prevj);
			uchar* datanext= foreground.ptr<uchar>(nextj);
			uchar* datalt= liantong.ptr<uchar>(j);
          for (int i=0; i<width; i++){  
				 int arr[8];
				 int previ=i-1,nexti=i+1;
				 if(i==0){previ=0;}
				 if(i==width-1){nexti=width-1;}
				unsigned char code = 0;  
				code |= (dataprev[previ] == 255) << 7;  
				code |= (dataprev[i] == 255) << 6;  
				code |= (dataprev[nexti] == 255) << 5;  
				code |= (data[nexti] == 255) << 4;
				code |= (datanext[nexti] == 255) << 3;  
				code |= (datanext[i] == 255) << 2;
				code |= (datanext[previ] == 255) << 1;  
				code |= (data[previ] == 255) << 0;
                datalt[i]=code; 
				}                    
		  }
		imshow("ws",liantong);
		//end extra2

		//腐蚀膨胀一下
		morphologyEx(foreground,foreground0,MORPH_CLOSE,Mat(3,3,CV_8U),Point(-1,-1),1);
        imshow("temp",foreground0);
		
		//精确找到运动物体
		front=IplImage(foreground0);
		stor = cvCreateMemStorage(0);
		cont = cvCreateSeq(CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint) , stor);
		 //找到所有轮廓
		cvFindContours(&front, stor, &cont, sizeof(CvContour), 
						CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));
		// 直接使用CONTOUR中的矩形来画轮廓
		for(;cont;cont = cont->h_next)
		{
				  CvRect r = ((CvContour*)cont)->rect;
				  if((r.height * r.width > CONTOUR_MAX_AERA)&&(r.x!=1)) // 面积小的方形抛弃掉
				     {
					  cvRectangle(&frame1, cvPoint(r.x,r.y),cvPoint(r.x + r.width, r.y + r.height),CV_RGB(255,0,0), 1, CV_AA,0);
					  mask2(Rect(r.x,r.y,r.width,r.height))=0;
					  		//part3 	
					  /*if(count>55){
							Mat dst(frame, Rect(r.x,r.y,r.width,r.height));
							imshow("dst",dst);
							imwrite("d:\\test.jpg",dst);
					  }*/
							
				  }
		}
		//cvShowImage("result",&frame1);

		//part3
		//做追踪
		imshow("mask2",mask2);
		if(count==0){
			gray_prev=gray.clone();
		}
		Mat feature=gray_prev.clone();

        for( int j =0; j < 240;j++ )
        {
          uchar* data= mask2.ptr<uchar>(j);  
		  uchar* featuredata= feature.ptr<uchar>(j); 
		  for (int i=0; i<320; i++) {  
			  featuredata[i]=(data[i]==255)?0:featuredata[i];
            }         
		}
		//找角点
		IplImage* img_feature=&IplImage(feature);
		IplImage* eig_img=cvCreateImage(cvGetSize(img_feature),IPL_DEPTH_32F,1);
		IplImage* temp_img=cvCloneImage(eig_img);
		const int MAX_CORNERS=1000;//定义角点个数最大值
		CvPoint2D32f* features_prev=new CvPoint2D32f[MAX_CORNERS];//分配保存角点的空间
		int corner_count=MAX_CORNERS;
		double quality_level=0.01;//or 0.01
		double min_distance=5;
		cvGoodFeaturesToTrack(img_feature,eig_img,temp_img,features_prev,&corner_count,quality_level,min_distance);

		//画角点
		for(int i=0;i<corner_count;i++)
		{
			//cvCircle(&img_feature,cvPoint((int)corners[i].x,(int)corners[i].y),1,CV_RGB(255,0,0),2,8);
			//cvCircle(&frame1,cvPoint((int)features_prev[i].x,(int)features_prev[i].y),1,CV_RGB(255,0,0),2,8);
		}

		//imshow("part3",feature);
		cvShowImage("part3",img_feature);

		//LK光流法
		char feature_found[ MAX_CORNERS ] ;
		float feature_errors[ MAX_CORNERS ] ;
		IplImage* IplGray=&IplImage(gray);
		IplImage* IplGray_prev;
		if(count==0){
			IplGray_prev=cvCloneImage(IplGray);
		}
		const int win_size = 10 ;
		CvSize pyr_sz = cvSize(IplGray->width + 8 ,IplGray->height / 3 ) ;
		CvSize img_sz = cvGetSize(IplGray);
		IplImage* pyr_prev = cvCreateImage(img_sz,IPL_DEPTH_32F,1) ;
		IplImage* pyr_cur = cvCreateImage(img_sz,IPL_DEPTH_32F,1) ;
		CvPoint2D32f*  features_cur = new CvPoint2D32f[ MAX_CORNERS ] ;

		cvCalcOpticalFlowPyrLK(
			IplGray_prev,
			IplGray,
			pyr_prev,
			pyr_cur,
			features_prev,
			features_cur,
			corner_count,
			cvSize(win_size,win_size),
			5,
			feature_found,
			feature_errors,
			cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER,20,0.3),
			0
			);
		double diffX=0,diffY=0;

		for ( int i = 0 ; i < corner_count ; i++)
		{
			if ( 0 == feature_found[i] || feature_errors[i] > 550 )
			{
				printf("error is %f \n" , feature_errors[i] ) ;
				continue ;
			}

			//printf("find it !\n") ;

			CvPoint pt_prev = cvPoint( features_prev[i].x , features_prev[i].y ) ;
			CvPoint pt_cur = cvPoint( features_cur[i].x , features_cur[i].y ) ;
			float dis=sqrt(pow((features_cur[i].y-features_prev[i].y),2)+pow((features_cur[i].x-features_prev[i].x),2));
			cvLine(&frame1,pt_prev,pt_cur,CV_RGB( 0,255,0),1);
			if(count==55){
				cout<<"x:"<<features_prev[i].x<<" y:"<<features_prev[i].y<<" x2:"<<features_cur[i].x<<" y2:"<<features_cur[i].y<<endl;
			}
			diffX=((features_cur[i].x-features_prev[i].x)+diffX*i)/float(i+1);
			diffY=((features_cur[i].y-features_prev[i].y)+diffY*i)/float(i+1);
		}
		if(count==55){
			cout<<"diffx:"<<diffX<<" diffy:"<<diffY<<endl;
		}
		IplGray_prev=cvCloneImage(IplGray);
		gray_prev=gray.clone();
		cvShowImage("result",&frame1);

		mask2(Rect(1,1,318,238))=255;

		count++;
		cout<<count<<endl;
		if(count>18){
			cvWaitKey(0);
		}
		

		// free memory
		cvReleaseMemStorage(&stor);

		if (waitKey(10) == 27)
		{
			stop = true;
		}
	}
}