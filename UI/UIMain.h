#pragma once

// For compilers that support precompilation, includes "wx/wx.h".
#include "wx/wxprec.h"
#include "wx/confbase.h"

#include "wx/wfstream.h"
#include "wx/zstream.h"
#include "wx/txtstrm.h"
#include "wx/glcanvas.h"

#include "reconUI.h"
#include "interop.h"
#include "../reconstruction/TomoRecon.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
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

class DTRMainWindow : public mainWindow{
protected:
	// Handlers for mainWindow events.
	void onNew(wxCommandEvent& event);
	void onOpen(wxCommandEvent& event);
	void onQuit(wxCommandEvent& event);
	void onAbout(wxCommandEvent& event);
	void onConfig(wxCommandEvent& event);

	wxPanel *DTRMainWindow::CreateNewPage() const;
public:
	// Constructor
	DTRMainWindow(wxWindow* parent);
	~DTRMainWindow();
};

class DTRConfigDialog : public configDialog{
protected:
	// Handlers for configDialog events.
	void onLoad(wxCommandEvent& event);
	void onSave(wxCommandEvent& event);
	void onOK(wxCommandEvent& event);
	void onCancel(wxCommandEvent& event);

public:
	/** Constructor */
	DTRConfigDialog(wxWindow* parent);
	~DTRConfigDialog();
};

// The OpenGL-enabled canvas
class CudaGLCanvas : public wxGLCanvas{
public:
	CudaGLCanvas(wxWindow *parent, wxWindowID id = wxID_ANY, int *gl_attrib = NULL, wxSize size = wxDefaultSize);

	virtual ~CudaGLCanvas();

	void OnPaint(wxPaintEvent& event);
	void OnChar(wxKeyEvent& event);
	void OnMouseEvent(wxMouseEvent& event);

private:
	cudaStream_t stream;
	cudaEvent_t  event;

	wxGLContext* m_glRC;
	interop* m_inter;

	wxDECLARE_NO_COPY_CLASS(CudaGLCanvas);
	wxDECLARE_EVENT_TABLE();
};

class GLFrame : public wxPanel {
public:
	GLFrame(wxAuiNotebook *frame,
		const wxPoint& pos = wxDefaultPosition,
		const wxSize& size = wxDefaultSize,
		long style = wxDEFAULT_FRAME_STYLE);

	virtual ~GLFrame();

	CudaGLCanvas *m_canvas;
};