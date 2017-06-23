
// GeometryDlg.cpp : implementation file
//

#include "stdafx.h"
#include "Geometry.h"
#include "GeometryDlg.h"
#include "afxdialogex.h"

#include <memory>


#ifdef _DEBUG
#define new DEBUG_NEW
#endif

using namespace std;

// CAboutDlg dialog used for App About

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

// Implementation
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CGeometryDlg dialog


CGeometryDlg::CGeometryDlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_GEOMETRY_DIALOG, pParent)
	, m_xsize(1915)
	, m_ysize(1440)
	, m_pixelThresholdLo(100)
	, m_pixelThresholdHi(350)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CGeometryDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_GeometryButton, GeometryButton);
	DDX_Control(pDX, IDC_PIXEL_THRESHOLD_LO, m_pixelThresholdLoCtrl);
	DDX_Text(pDX, IDC_PIXEL_THRESHOLD_LO, m_pixelThresholdLo);
	DDX_Control(pDX, IDC_PIXEL_THRESHOLD_HI, m_pixelThresholdHiCtrl);
	DDX_Text(pDX, IDC_PIXEL_THRESHOLD_HI, m_pixelThresholdHi);
	DDX_Control(pDX, IDC_XSIZE, m_XsizeCtrl);
	DDX_Text(pDX, IDC_XSIZE, m_xsize);
	DDX_Control(pDX, IDC_YSIZE, m_YsizeCtrl);
	DDX_Text(pDX, IDC_YSIZE, m_ysize);
}

BEGIN_MESSAGE_MAP(CGeometryDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_GeometryButton, &CGeometryDlg::OnClickedGeometryButton)
	ON_EN_UPDATE(IDC_XSIZE, &CGeometryDlg::OnChangeXsize)
	ON_EN_UPDATE(IDC_YSIZE, &CGeometryDlg::OnChangeYsize)
	ON_EN_UPDATE(IDC_PIXEL_THRESHOLD_LO, OnChangePixelThresholdLo)
	ON_EN_UPDATE(IDC_PIXEL_THRESHOLD_HI, OnChangePixelThresholdHi)
END_MESSAGE_MAP()


// CGeometryDlg message handlers

BOOL CGeometryDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// Add "About..." menu item to system menu.

	// IDM_ABOUTBOX must be in the system command range.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	// TODO: Add extra initialization here

	m_pixelThresholdLo = 100;
	m_pixelThresholdHi = 350;

	m_xsize = 1915;
	m_ysize = 1440;

	return TRUE;  // return TRUE  unless you set the focus to a control
}

void CGeometryDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CGeometryDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CGeometryDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CGeometryDlg::OnClickedGeometryButton()
{
	INT_PTR dialogStatus = 0;
	int status = 0;

	CString filename, dirname;

	//this is the expected filename based on images acquired with UNC system naming conventions
	CString extension = _T(".raw");
	CString basename = _T("AcquiredImage1_");
	CString divider = _T("\\");

	// 1st parameter: sending false does save as
	// sending true to this call does open
	CFileDialog fileDlg(TRUE);

	//open file dialog to select first image in sequence to use for geometry measurement
	dialogStatus = fileDlg.DoModal();

	// if select file, get filename information
	if (dialogStatus == IDOK)
	{
		filename = fileDlg.GetPathName(); // return full path and filename
		dirname = fileDlg.GetFolderPath();
	}
	else if (dialogStatus == IDCANCEL)
	{
		exit(0);
	}
}

void CGeometryDlg::OnChangePixelThresholdLo()
{
	CString txt;

	m_pixelThresholdLoCtrl.GetWindowText(txt);

	m_pixelThresholdLo = _ttoi(txt);
}

void CGeometryDlg::OnChangePixelThresholdHi()
{
	CString txt;

	m_pixelThresholdHiCtrl.GetWindowText(txt);

	m_pixelThresholdHi = _ttoi(txt);
}

void CGeometryDlg::OnChangeXsize()
{
	CString txt;

	m_XsizeCtrl.GetWindowText(txt);

	m_xsize = _ttoi(txt);
}

void CGeometryDlg::OnChangeYsize()
{
	CString txt;

	m_YsizeCtrl.GetWindowText(txt);

	m_ysize = _ttoi(txt);
}
