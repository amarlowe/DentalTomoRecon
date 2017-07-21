///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version Jun 17 2015)
// http://www.wxformbuilder.org/
//
// PLEASE DO "NOT" EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#ifndef __RECONUI_H__
#define __RECONUI_H__

#include <wx/artprov.h>
#include <wx/xrc/xmlres.h>
#include <wx/string.h>
#include <wx/bitmap.h>
#include <wx/image.h>
#include <wx/icon.h>
#include <wx/menu.h>
#include <wx/gdicmn.h>
#include <wx/font.h>
#include <wx/colour.h>
#include <wx/settings.h>
#include <wx/textctrl.h>
#include <wx/sizer.h>
#include <wx/panel.h>
#include <wx/aui/auibook.h>
#include <wx/statusbr.h>
#include <wx/frame.h>
#include <wx/aui/aui.h>
#include <wx/gauge.h>
#include <wx/dialog.h>
#include <wx/stattext.h>
#include <wx/radiobox.h>
#include <wx/grid.h>
#include <wx/button.h>

///////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
/// Class mainWindow
///////////////////////////////////////////////////////////////////////////////
class mainWindow : public wxFrame 
{
	private:
	
	protected:
		wxMenuBar* m_menubar1;
		wxMenu* file;
		wxMenu* config;
		wxMenu* reconstruction;
		wxMenu* help;
		wxPanel* m_panel10;
		wxStatusBar* m_statusBar1;
		
		// Virtual event handlers, overide them in your derived class
		virtual void onKeyDown( wxKeyEvent& event ) { event.Skip(); }
		virtual void onKeyUp( wxKeyEvent& event ) { event.Skip(); }
		virtual void onNew( wxCommandEvent& event ) { event.Skip(); }
		virtual void onOpen( wxCommandEvent& event ) { event.Skip(); }
		virtual void onSave( wxCommandEvent& event ) { event.Skip(); }
		virtual void onQuit( wxCommandEvent& event ) { event.Skip(); }
		virtual void onConfig( wxCommandEvent& event ) { event.Skip(); }
		virtual void onGainSelect( wxCommandEvent& event ) { event.Skip(); }
		virtual void onDarkSelect( wxCommandEvent& event ) { event.Skip(); }
		virtual void onContinue( wxCommandEvent& event ) { event.Skip(); }
		virtual void onContRun( wxCommandEvent& event ) { event.Skip(); }
		virtual void onStep( wxCommandEvent& event ) { event.Skip(); }
		virtual void onAbout( wxCommandEvent& event ) { event.Skip(); }
		virtual void onPageChange( wxAuiNotebookEvent& event ) { event.Skip(); }
		
	
	public:
		wxAuiNotebook* m_auinotebook6;
		wxTextCtrl* m_textCtrl8;
		
		mainWindow( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Tomogrophy Reconstruction"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1074,681 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );
		wxAuiManager m_mgr;
		
		~mainWindow();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class RunBox
///////////////////////////////////////////////////////////////////////////////
class RunBox : public wxDialog 
{
	private:
	
	protected:
	
	public:
		wxGauge* m_gauge2;
		
		RunBox( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Reconstruction running"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 208,61 ), long style = wxDEFAULT_DIALOG_STYLE|wxSTAY_ON_TOP ); 
		~RunBox();
	
};

///////////////////////////////////////////////////////////////////////////////
/// Class configDialog
///////////////////////////////////////////////////////////////////////////////
class configDialog : public wxFrame 
{
	private:
	
	protected:
		wxStaticText* m_staticText1;
		wxRadioBox* generateDistance;
		wxStaticText* m_staticText2;
		wxTextCtrl* estimatedDistance;
		wxStaticText* m_staticText4;
		wxTextCtrl* reconstructionSlices;
		wxStaticText* m_staticText3;
		wxTextCtrl* sliceThickness;
		wxStaticText* m_staticText7;
		wxRadioBox* edgeBlurEnabled;
		wxStaticText* m_staticText8;
		wxRadioBox* denosingEnabled;
		wxStaticText* m_staticText6;
		wxRadioBox* orientation;
		wxStaticText* m_staticText5;
		wxRadioBox* rotationEnabled;
		wxStaticText* m_staticText13;
		wxStaticText* m_staticText9;
		wxTextCtrl* pixelWidth;
		wxStaticText* m_staticText10;
		wxTextCtrl* pixelHeight;
		wxStaticText* m_staticText11;
		wxTextCtrl* pitchHeight;
		wxStaticText* m_staticText12;
		wxTextCtrl* pitchWidth;
		wxStaticText* m_staticText14;
		wxGrid* m_grid1;
		wxButton* loadConfig;
		wxButton* saveConfig;
		wxButton* ok;
		wxButton* cancel;
		
		// Virtual event handlers, overide them in your derived class
		virtual void onLoad( wxCommandEvent& event ) { event.Skip(); }
		virtual void onSave( wxCommandEvent& event ) { event.Skip(); }
		virtual void onOK( wxCommandEvent& event ) { event.Skip(); }
		virtual void onCancel( wxCommandEvent& event ) { event.Skip(); }
		
	
	public:
		
		configDialog( wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& title = wxT("Reconstruction Configuration"), const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxSize( 1343,511 ), long style = wxDEFAULT_FRAME_STYLE|wxTAB_TRAVERSAL );
		
		~configDialog();
	
};

#endif //__RECONUI_H__
