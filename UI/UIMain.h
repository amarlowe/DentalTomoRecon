#pragma once

// For compilers that support precompilation, includes "wx/wx.h".
#include "wx/wxprec.h"
#include "wx/confbase.h"
#include "reconUI.h"

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

class DTRMainWindow : public mainWindow
{
protected:
	// Handlers for mainWindow events.
	void onOpen(wxCommandEvent& event);
	void onQuit(wxCommandEvent& event);
	void onAbout(wxCommandEvent& event);
	void onConfig(wxCommandEvent& event);
public:
	// Constructor
	DTRMainWindow(wxWindow* parent);
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
};