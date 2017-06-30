#pragma once

// For compilers that support precompilation, includes "wx/wx.h".
#include "wx/wxprec.h"
#include "wx/confbase.h"
#include "reconUI.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include "wx/glcanvas.h"
#include "wx/wfstream.h"
#include "wx/zstream.h"
#include "wx/txtstrm.h"

// the maximum number of vertex in the loaded .dat file
#define MAXVERTS     10000

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

// global options which can be set through command-line options
GLboolean g_use_vertex_arrays = GL_FALSE;
GLboolean g_doubleBuffer = GL_TRUE;
GLboolean g_smooth = GL_TRUE;
GLboolean g_lighting = GL_TRUE;

class DTRMainWindow : public mainWindow
{
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

class DTRConfigDialog : public configDialog
{
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
class TestGLCanvas : public wxGLCanvas
{
public:
	TestGLCanvas(wxWindow *parent,
		wxWindowID id = wxID_ANY,
		int *gl_attrib = NULL,
		wxSize size = wxDefaultSize);

	virtual ~TestGLCanvas();

	void OnPaint(wxPaintEvent& event);
	void OnChar(wxKeyEvent& event);
	void OnMouseEvent(wxMouseEvent& event);

	void LoadSurface(const wxString& filename);
	void InitMaterials();
	void InitGL();

private:
	wxGLContext* m_glRC;

	GLfloat m_verts[MAXVERTS][3];
	GLfloat m_norms[MAXVERTS][3];
	GLint m_numverts;

	GLfloat m_xrot;
	GLfloat m_yrot;

	wxDECLARE_NO_COPY_CLASS(TestGLCanvas);
	wxDECLARE_EVENT_TABLE();
};

class GLFrame : public wxPanel {
public:
	GLFrame(wxAuiNotebook *frame,
		const wxPoint& pos = wxDefaultPosition,
		const wxSize& size = wxDefaultSize,
		long style = wxDEFAULT_FRAME_STYLE);

	virtual ~GLFrame();

	TestGLCanvas *m_canvas;
};