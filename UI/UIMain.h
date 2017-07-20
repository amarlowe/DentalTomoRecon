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

bool first = true;

class DTRConfigDialog : public configDialog {
protected:
	// Handlers for configDialog events.
	void onLoad(wxCommandEvent& event);
	void onSave(wxCommandEvent& event);
	void onOK(wxCommandEvent& event);
	void onCancel(wxCommandEvent& event);

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

class DTRMainWindow : public mainWindow{
protected:
	// Generate a System object from config file
	TomoError genSys(struct SystemControl * Sys);

	// Handlers for mainWindow events.
	void onNew(wxCommandEvent& event);
	void onOpen(wxCommandEvent& event);
	void onSave(wxCommandEvent& event);
	void onQuit(wxCommandEvent& event);
	void onAbout(wxCommandEvent& event);
	void onConfig(wxCommandEvent& event);
	void onGainSelect(wxCommandEvent& event);
	void onDarkSelect(wxCommandEvent& event);
	void onStep(wxCommandEvent& event);
	void onContinue(wxCommandEvent& event);
	void onPageChange(wxCommandEvent& event);

	//constant globals
	const int NumViews = NUMVIEWS;

public:
	// Constructor
	DTRMainWindow(wxWindow* parent);
	~DTRMainWindow();

	DTRConfigDialog* cfgDialog = NULL;
	wxPanel *DTRMainWindow::CreateNewPage();
	void onContRun(wxCommandEvent& event);

	//User generated filenames
	wxString gainFilepath;
	wxString darkFilepath;
};

// The OpenGL-enabled canvas
class CudaGLCanvas : public wxGLCanvas{
public:
	CudaGLCanvas(wxWindow *parent, struct SystemControl * Sys, wxString gainFile, wxString darkFile, wxWindowID id = wxID_ANY, int *gl_attrib = NULL, wxSize size = wxDefaultSize);

	virtual ~CudaGLCanvas();

	void OnEvent(wxCommandEvent& event);
	void OnPaint(wxPaintEvent& event);
	void OnChar(wxKeyEvent& event);
	void OnMouseEvent(wxMouseEvent& event);
	void OnScroll(int index);

	void paint();

	TomoRecon* recon;

private:
	int imageIndex = 0;
	int reconIndex = 0;

	wxGLContext* m_glRC;

	wxDECLARE_NO_COPY_CLASS(CudaGLCanvas);
	wxDECLARE_EVENT_TABLE();
};

class GLFrame : public wxPanel {
public:
	GLFrame(wxAuiNotebook *frame, struct SystemControl * Sys, wxString gainFile, wxString darkFile,
		const wxPoint& pos = wxDefaultPosition,
		const wxSize& size = wxDefaultSize,
		long style = wxDEFAULT_FRAME_STYLE);

	virtual ~GLFrame();

	void OnScroll(wxScrollEvent& event);
	void OnMousewheel(wxMouseEvent& event);

	CudaGLCanvas *m_canvas;
	wxScrollBar* m_scrollBar;

	wxDECLARE_NO_COPY_CLASS(GLFrame);
	wxDECLARE_EVENT_TABLE();
};

BEGIN_DECLARE_EVENT_TYPES()
DECLARE_EVENT_TYPE(PAINT_IT, -1)
END_DECLARE_EVENT_TYPES()
class ReconThread : public wxThread{
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