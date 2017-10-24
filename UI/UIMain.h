#pragma once

// For compilers that support precompilation, includes "wx/wx.h".
#include "wx/wxprec.h"
#include "wx/confbase.h"

#include "wx/wfstream.h"
#include "wx/zstream.h"
#include "wx/txtstrm.h"
#include "wx/glcanvas.h"
#include <wx/thread.h>

#include "reconUI.h"
#include "../reconstruction/TomoRecon.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include "../fileIO/cJSON.h"
//#include <GL/freeglut.h>

#include "dcmtk/config/osconfig.h" 
#include "dcmtk/dcmdata/dctk.h"
#include "dcmtk/dcmimgle/dcmimage.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

// for all others, include the necessary headers (this file is usually all you
// need because it includes almost all "standard" wxWidgets headers)
#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif

// the application icon (under Windows it is in resources and even
// though we could still include the XPM here it would be unused)
#ifndef wxHAS_IMAGES_IN_RESOURCES
#include "../sample.xpm"
#endif

#define MOUSEWHEELMAG 120
#define SCROLLFACTOR 10
#define ENHANCEFACTOR 100.0f
#define WINLVLFACTOR 255
#define STEPFACTOR	10.0f
#define SCANFACTOR 100.0f
#define MAXSLICE 150

//define private tags for saving program settings
#define PRIVATE_CREATOR_NAME "Xinvivo"

#define PRIVATE_CREATOR_TAG		0x0029, 0x0010
#define PRIVATE_STEP_TAG		0x0029, 0x1000
#define PRIVATE_DERDIS_TAG		0x0029, 0x1001
#define PRIVATE_EDGERAT_TAG		0x0029, 0x1002
#define PRIVATE_DATADIS_TAG		0x0029, 0x1003
#define PRIVATE_HORFLIP_TAG		0x0029, 0x1004
#define PRIVATE_VERFLIP_TAG		0x0029, 0x1005
#define PRIVATE_LOGVIEW_TAG		0x0029, 0x1006
#define PRIVATE_SCVEREN_TAG		0x0029, 0x1007
#define PRIVATE_SCVERVL_TAG		0x0029, 0x1008
#define PRIVATE_SCHOREN_TAG		0x0029, 0x1009
#define PRIVATE_SCHORVL_TAG		0x0029, 0x100a
#define PRIVATE_OUTNSEN_TAG		0x0029, 0x100b
#define PRIVATE_OUTNSVL_TAG		0x0029, 0x100c
#define PRIVATE_TVEN_TAG		0x0029, 0x100d
#define PRIVATE_TVLMDA_TAG		0x0029, 0x100e
#define PRIVATE_TVITER_TAG		0x0029, 0x100f
#define PRIVATE_DISSRT_TAG		0x0029, 0x1010
#define PRIVATE_DISEND_TAG		0x0029, 0x1011
#define PRIVATE_USEGN_TAG		0x0029, 0x1012

#define PRV_PrivateCreator  DcmTag(PRIVATE_CREATOR_TAG)
#define PRV_StepSize		DcmTag(PRIVATE_STEP_TAG, PRIVATE_CREATOR_NAME)
#define PRV_DerDisplay		DcmTag(PRIVATE_DERDIS_TAG, PRIVATE_CREATOR_NAME)
#define PRV_EdgeRatio		DcmTag(PRIVATE_EDGERAT_TAG, PRIVATE_CREATOR_NAME)
#define PRV_DataDisplay		DcmTag(PRIVATE_DATADIS_TAG, PRIVATE_CREATOR_NAME)
#define PRV_HorFlip			DcmTag(PRIVATE_HORFLIP_TAG, PRIVATE_CREATOR_NAME)
#define PRV_VertFlip		DcmTag(PRIVATE_VERFLIP_TAG, PRIVATE_CREATOR_NAME)
#define PRV_LogView			DcmTag(PRIVATE_LOGVIEW_TAG, PRIVATE_CREATOR_NAME)
#define PRV_ScanVertEn		DcmTag(PRIVATE_SCVEREN_TAG, PRIVATE_CREATOR_NAME)
#define PRV_ScanVertVal		DcmTag(PRIVATE_SCVERVL_TAG, PRIVATE_CREATOR_NAME)
#define PRV_ScanHorEn		DcmTag(PRIVATE_SCHOREN_TAG, PRIVATE_CREATOR_NAME)
#define PRV_ScanHorVal		DcmTag(PRIVATE_SCHORVL_TAG, PRIVATE_CREATOR_NAME)
#define PRV_OutNoiseEn		DcmTag(PRIVATE_OUTNSEN_TAG, PRIVATE_CREATOR_NAME)
#define PRV_OutNoiseMax		DcmTag(PRIVATE_OUTNSVL_TAG, PRIVATE_CREATOR_NAME)
#define PRV_TVEn			DcmTag(PRIVATE_TVEN_TAG, PRIVATE_CREATOR_NAME)
#define PRV_TVLambda		DcmTag(PRIVATE_TVLMDA_TAG, PRIVATE_CREATOR_NAME)
#define PRV_TVIter			DcmTag(PRIVATE_TVITER_TAG, PRIVATE_CREATOR_NAME)
#define PRV_DisStart		DcmTag(PRIVATE_DISSRT_TAG, PRIVATE_CREATOR_NAME)
#define PRV_DisEnd			DcmTag(PRIVATE_DISEND_TAG, PRIVATE_CREATOR_NAME)
#define PRV_UseGain			DcmTag(PRIVATE_USEGN_TAG, PRIVATE_CREATOR_NAME)

typedef enum {
	Status = 0,
	xOffset,
	yOffset
} status_t;

typedef enum {
	box1,
	box2,
	lower,
	upper
} input_t;

bool first = true;

class DTRSliceSave : public sliceDialog {
protected:
	void onSliceValue(wxCommandEvent& event);
public:
	DTRSliceSave(wxWindow* parent);
	~DTRSliceSave();

	long value = 0;
};

class ReconCon : public reconConfig {
protected:
	wxPanel* drawPanel;

	void onDistance(wxCommandEvent& event);
	void onOk(wxCommandEvent& event);
	void onCancel(wxCommandEvent& event);
	void onSetStartDis(wxCommandEvent& event);
	void onSetEndDis(wxCommandEvent& event);
	void onStepSlider(wxScrollEvent& event);
	void onClose(wxCloseEvent& event);
	void onEnableGain(wxCommandEvent& event);

	void onToolbarChoice(wxCommandEvent& event);

	//Scanline correction
	void onScanVertEnable(wxCommandEvent& event);
	void onScanVert(wxScrollEvent& event);
	void onResetScanVert(wxCommandEvent& event);
	void onScanHorEnable(wxCommandEvent& event);
	void onScanHor(wxScrollEvent& event);
	void onResetScanHor(wxCommandEvent& event);

	//Denoising
	void onNoiseMaxEnable(wxCommandEvent& event);
	void onNoiseMax(wxScrollEvent& event);
	void onResetNoiseMax(wxCommandEvent& event);
	void onTVEnable(wxCommandEvent& event);
	void onResetLambda(wxCommandEvent& event);
	void onLambdaSlider(wxScrollEvent& event);
	void onResetIter(wxCommandEvent& event);
	void onIterSlider(wxScrollEvent& event);
public:
	/** Constructor */
	ReconCon(wxWindow* parent, wxString filename, wxString gainFile);
	~ReconCon();

	void setValues();

	double startDis = 0.0f;
	double endDis = 10.0f;

	wxString filename;
	wxString gainFilepath;
	float stepSize;
	bool scanVertIsEnabled;
	bool scanHorIsEnabled;
	float scanVertVal;
	float scanHorVal;
	bool noiseMaxIsEnabled;
	int noiseMaxValue;
	bool TVIsEnabled;
	int TVLambdaVal;
	int TVIterVal;
	bool gainIsEnabled;
	bool canceled = true;
};

class DTRConfigDialog : public configDialog {
protected:
	// Handlers for configDialog events.
	void onLoad(wxCommandEvent& event);
	void onSave(wxCommandEvent& event);
	void onOK(wxCommandEvent& event);
	void onCancel(wxCommandEvent& event);
	void onConfigChar(wxCommandEvent& event);

	TomoError ParseLegacyTxt(std::string FilePath);
	TomoError ParseJSONFile(std::string FilePath);
	TomoError checkInputs();

	//User generated filenames
	std::string configFilepath;

public:
	/** Constructor */
	DTRConfigDialog(wxWindow* parent);
	~DTRConfigDialog();
};

// The OpenGL-enabled canvas
class CudaGLCanvas : public wxGLCanvas{
public:
	CudaGLCanvas(wxWindow *parent, wxStatusBar* status, struct SystemControl * Sys,
		wxWindowID id = wxID_ANY, int *gl_attrib = NULL, wxSize size = wxDefaultSize);

	virtual ~CudaGLCanvas();

	void OnPaint(wxPaintEvent& event);
	void OnChar(wxKeyEvent& event);
	void OnMouseEvent(wxMouseEvent& event);
	void OnScroll(int index);

	void paint(bool disChanged = false, wxTextCtrl* dis = NULL, wxSlider* zoom = NULL, wxStaticText* zLbl = NULL,
		wxSlider* window = NULL, wxStaticText* wLbl = NULL, wxSlider* level = NULL, wxStaticText* lLbl = NULL);

	TomoRecon* recon;
	wxStatusBar* m_status = NULL;
	wxTextCtrl* distanceControl = NULL;
	wxSlider* zoomSlider = NULL;
	wxStaticText* zoomLabel = NULL;
	wxSlider* windowSlider = NULL;
	wxStaticText* windowLabel = NULL;
	wxSlider* levelSlider = NULL;
	wxStaticText* levelLabel = NULL;

	TomoError launchError;

private:
	int imageIndex = 0;
	int reconIndex = 0;

	wxGLContext* m_glRC;

	wxDECLARE_NO_COPY_CLASS(CudaGLCanvas);
	wxDECLARE_EVENT_TABLE();
};

class GLFrame : public wxPanel {
public:
	GLFrame(wxWindow *frame, struct SystemControl * Sys, wxString filename,
		wxStatusBar* status = NULL,
		const wxPoint& pos = wxDefaultPosition,
		const wxSize& size = wxDefaultSize,
		long style = wxDEFAULT_FRAME_STYLE);

	virtual ~GLFrame();

	void OnScroll(wxScrollEvent& event);
	void OnMousewheel(wxMouseEvent& event);
	void hideScrollBar();
	void showScrollBar(int steps, int current);

	CudaGLCanvas *m_canvas;
	wxScrollBar* m_scrollBar;
	wxStatusBar* m_status;
	wxBoxSizer* bSizer = NULL;

	wxString filename;

	wxDECLARE_NO_COPY_CLASS(GLFrame);
	wxDECLARE_EVENT_TABLE();
};

// The OpenGL-enabled canvas
class CudaGLInCanvas : public wxGLCanvas {
public:
	CudaGLInCanvas(wxWindow *parent, bool vertical, struct SystemControl * Sys, wxString gainFile, wxString filename,
		wxWindowID id = wxID_ANY, int *gl_attrib = NULL, wxSize size = wxDefaultSize);

	virtual ~CudaGLInCanvas();

	void OnPaint(wxPaintEvent& event);
	void OnMouseEvent(wxMouseEvent& event);
	void OnChar(wxKeyEvent& event);

	void paint();

	TomoRecon* recon;
	input_t state = box1;

	int x1, x2, y1, y2;

private:
	int imageIndex = 0;
	int reconIndex = 0;

	wxGLContext* m_glRC;

	wxDECLARE_NO_COPY_CLASS(CudaGLInCanvas);
	wxDECLARE_EVENT_TABLE();
};

class GLWindow : public wxDialog {
public:
	GLWindow(wxWindow *parent, bool vertical, struct SystemControl * Sys, wxString gainFile, wxString filename,
		const wxPoint& pos = wxDefaultPosition,
		const wxSize& size = wxDefaultSize,
		long style = wxDEFAULT_FRAME_STYLE);

	virtual ~GLWindow();

	void OnMousewheel(wxMouseEvent& event);
	void onClose(wxCloseEvent& event);

	CudaGLInCanvas *m_canvas;

	wxDECLARE_NO_COPY_CLASS(GLWindow);
	wxDECLARE_EVENT_TABLE();
};

class DTRResDialog : public resDialog {
protected:
	// Handlers for configDialog events.
	void onAddNew(wxCommandEvent& event);
	void onRemove(wxCommandEvent& event);
	void onOk(wxCommandEvent& event);
	void onCancel(wxCommandEvent& event);

	GLWindow* frame;

public:
	/** Constructor */
	DTRResDialog(wxWindow* parent);
	~DTRResDialog();
};

class DTRMainWindow : public mainWindow {
protected:
	//helpers
	bool checkForConsole();
	derivative_t getEnhance();
	TomoError launchReconConfig(TomoRecon * recon, wxString filename);
	void refreshToolbars(GLFrame* currentFrame);

	// Handlers for mainWindow events.
	void onNew(wxCommandEvent& event);
	void onOpen(wxCommandEvent& event);
	void onSave(wxCommandEvent& event);
	void onExportRecon(wxCommandEvent& event);
	void onQuit(wxCommandEvent& event);
	void onAbout(wxCommandEvent& event);
	void onConfig(wxCommandEvent& event);
	void onGainSelect(wxCommandEvent& event);
	void onReconSetup(wxCommandEvent& event);
	void onResetFocus(wxCommandEvent& event);
	void onResList(wxCommandEvent& event);
	void onContList(wxCommandEvent& event);
	void onRunTest(wxCommandEvent& event);
	void onTestGeo(wxCommandEvent& event);
	void onAutoGeo(wxCommandEvent& event);
	void onPageChange(wxAuiNotebookEvent& event);
	void onPageClose(wxAuiNotebookEvent& event);

	//Toolbar functions

	//Navigation
	void onDistance(wxCommandEvent& event);
	void onAutoFocus(wxCommandEvent& event);
	void onAutoLight(wxCommandEvent& event);
	void onWindowSlider(wxScrollEvent& event);
	void onLevelSlider(wxScrollEvent& event);
	void onZoomSlider(wxScrollEvent& event);
	void onAutoAll(wxCommandEvent& event);
	void onVertFlip(wxCommandEvent& event);
	void onHorFlip(wxCommandEvent& event);
	void onLogView(wxCommandEvent& event);
	void onDataDisplay(wxCommandEvent& event);

	//Edge enhancement
	void onToolbarChoice(wxCommandEvent& event);
	void onXEnhance(wxCommandEvent& event);
	void onYEnhance(wxCommandEvent& event);
	void onAbsEnhance(wxCommandEvent& event);
	void onResetEnhance(wxCommandEvent& event);
	void onEnhanceRatio(wxScrollEvent& event);

	//constant globals
	const int NumViews = NUMVIEWS;
	int runIterations;

public:
	// Generate a System object from config file
	TomoError genSys(struct SystemControl * Sys);
	void setDataDisplay(GLFrame* currentFrame, sourceData selection);

	void onContinuous();

	// Constructor
	DTRMainWindow(wxWindow* parent);
	~DTRMainWindow();

	DTRConfigDialog* cfgDialog = NULL;
	wxPanel * CreateNewPage(wxString filename);

	//User generated filenames
	wxString gainFilepath;
};

BEGIN_DECLARE_EVENT_TYPES()
DECLARE_EVENT_TYPE(PAINT_IT, -1)
END_DECLARE_EVENT_TYPES()
class ReconThread : public wxThread {
public:
	ReconThread(wxEvtHandler* pParent, TomoRecon* recon, GLFrame* Frame, wxStatusBar* status, wxTextCtrl* m_textCtrl);
private:
	wxEvtHandler* m_pParent;
	TomoRecon* m_recon;
	GLFrame* currentFrame;
	wxStatusBar* status;
	wxTextCtrl* m_textCtrl;

	ExitCode Entry();
};

class saveThread : public wxThread {
public:
	saveThread(TomoRecon* recon, wxStatusBar* status);
private:
	TomoRecon* m_recon;
	wxStatusBar* status;

	ExitCode Entry();
};
