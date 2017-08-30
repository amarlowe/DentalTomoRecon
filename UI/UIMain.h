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
#define ENHANCEDEFAULT 5.0f
#define SCANVERTDEFAULT 0.25f
#define SCANHORDEFAULT 0.1f
#define NOISEMAXDEFAULT 700
#define ENHANCEFACTOR 10.0f
#define WINLVLFACTOR 255
#define STEPFACTOR	10.0f
#define SCANFACTOR 100.0f
#define MAXSLICE 150
#define LAMBDADEFAULT 2
#define ITERDEFAULT 20

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
	CudaGLCanvas(wxWindow *parent, wxStatusBar* status, struct SystemControl * Sys, wxString gainFile, wxString filename, 
		wxWindowID id = wxID_ANY, int *gl_attrib = NULL, wxSize size = wxDefaultSize);

	virtual ~CudaGLCanvas();

	void OnPaint(wxPaintEvent& event);
	void OnChar(wxKeyEvent& event);
	void OnMouseEvent(wxMouseEvent& event);
	void OnScroll(int index);

	void paint(bool disChanged = false, wxTextCtrl* dis = NULL, wxSlider* zoom = NULL, wxStaticText* zLbl = NULL,
		wxSlider* window = NULL, wxStaticText* wLbl = NULL, wxSlider* level = NULL, wxStaticText* lLbl = NULL);

	TomoRecon* recon;
	wxStatusBar* m_status;
	wxTextCtrl* distanceControl;
	wxSlider* zoomSlider;
	wxStaticText* zoomLabel;
	wxSlider* windowSlider;
	wxStaticText* windowLabel;
	wxSlider* levelSlider;
	wxStaticText* levelLabel;

private:
	int imageIndex = 0;
	int reconIndex = 0;

	wxGLContext* m_glRC;

	wxDECLARE_NO_COPY_CLASS(CudaGLCanvas);
	wxDECLARE_EVENT_TABLE();
};

class GLFrame : public wxPanel {
public:
	GLFrame(wxAuiNotebook *frame, wxStatusBar* status, struct SystemControl * Sys, wxString gainFile, wxString filename,
		const wxPoint& pos = wxDefaultPosition,
		const wxSize& size = wxDefaultSize,
		long style = wxDEFAULT_FRAME_STYLE);

	virtual ~GLFrame();

	void OnScroll(wxScrollEvent& event);
	void OnMousewheel(wxMouseEvent& event);
	void hideScrollBar();
	void showScrollBar();

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

	// Handlers for mainWindow events.
	void onNew(wxCommandEvent& event);
	void onOpen(wxCommandEvent& event);
	void onSave(wxCommandEvent& event);
	void onQuit(wxCommandEvent& event);
	void onAbout(wxCommandEvent& event);
	void onConfig(wxCommandEvent& event);
	void onGainSelect(wxCommandEvent& event);
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
	void onStepSlider(wxScrollEvent& event);
	void onAutoLight(wxCommandEvent& event);
	void onWindowSlider(wxScrollEvent& event);
	void onLevelSlider(wxScrollEvent& event);
	void onZoomSlider(wxScrollEvent& event);
	void onAutoAll(wxCommandEvent& event);
	void onVertFlip(wxCommandEvent& event);
	void onHorFlip(wxCommandEvent& event);
	void onLogView(wxCommandEvent& event);
	void onProjectionView(wxCommandEvent& event);

	//Edge enhancement
	void onToolbarChoice(wxCommandEvent& event);
	void onXEnhance(wxCommandEvent& event);
	void onYEnhance(wxCommandEvent& event);
	void onAbsEnhance(wxCommandEvent& event);
	void onResetEnhance(wxCommandEvent& event);
	void onEnhanceRatio(wxScrollEvent& event);

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

	//constant globals
	const int NumViews = NUMVIEWS;

public:
	// Generate a System object from config file
	TomoError genSys(struct SystemControl * Sys);

	void onContinuous();

	// Constructor
	DTRMainWindow(wxWindow* parent);
	~DTRMainWindow();

	DTRConfigDialog* cfgDialog = NULL;
	DTRResDialog* resDialog = NULL;
	wxPanel * CreateNewPage(wxString filename);

	//User generated filenames
	wxString gainFilepath;
};