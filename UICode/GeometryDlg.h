
// GeometryDlg.h : header file
//

#pragma once
#include "afxwin.h"

#include <string>
#include <iostream>
#include <fstream>

using namespace std;

//the opencv software finds all possible "objects" even if not an object
//limit the number of objects to 3000, 
//there should only by the number objects as rods in the phantom
//if more than 3000, just stop
//if not properly aligned, may get many more, hopefully, can improve
//process so always aligned properly but for now, with prototypes,
//need to account for extraneous objects
#define MAX_NUM_OBJECTS		3000


// CGeometryDlg dialog
class CGeometryDlg : public CDialogEx
{
// Construction
public:
	CGeometryDlg(CWnd* pParent = NULL);	// standard constructor

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_GEOMETRY_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


// Implementation
protected:
	HICON m_hIcon;

	std::ofstream m_foutdetailstxt;

	//valid component array
	bool m_validcomponent[MAX_NUM_OBJECTS];

	int m_actualobjcnt;

	//left point array
	float m_lefty[MAX_NUM_OBJECTS];

	int m_xsize;
	int m_ysize;

	int FindIntersectingLines();

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	CButton GeometryButton;
	afx_msg void OnClickedGeometryButton();
	afx_msg void OnChangePixelThresholdLo();
	afx_msg void OnChangePixelThresholdHi();
	CEdit m_pixelThresholdLoCtrl;
	CEdit m_pixelThresholdHiCtrl;
	CEdit m_XsizeCtrl;
	CEdit m_YsizeCtrl;
	afx_msg void OnChangeXsize();
	afx_msg void OnChangeYsize();
	int m_pixelThresholdLo;
	int m_pixelThresholdHi;
};
