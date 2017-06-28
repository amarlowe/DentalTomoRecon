#include "UIMain.h"

// Define a new application type, each program should derive a class from wxApp
class MyApp : public wxApp
{
public:
	virtual bool OnInit() wxOVERRIDE;
};

//Thread entry point, causes onInit to run like a main function would.
wxIMPLEMENT_APP(MyApp);

//Main equivalent: the program execution starts here
bool MyApp::OnInit()
{
	//set name for config files in registry
	SetVendorName(wxT("Xinvivo"));

	// call the base class initialization method, parses command line inputs
	if (!wxApp::OnInit())
		return false;

	// create the main application window
	DTRMainWindow *frame = new DTRMainWindow(NULL);
	frame->Show(true);

	return true;
}

// ----------------------------------------------------------------------------
// main frame
// ----------------------------------------------------------------------------

DTRMainWindow::DTRMainWindow(wxWindow* parent) : mainWindow(parent){
	//empty constructor, mainWindow handles everything
}

// event handlers
void DTRMainWindow::onOpen(wxCommandEvent& WXUNUSED(event)) {
	wxMessageBox(wxT("TODO"),
		wxT("TODO"),
		wxICON_INFORMATION | wxOK);
}

void DTRMainWindow::onQuit(wxCommandEvent& WXUNUSED(event)){
	// true is to force the frame to close
	Close(true);
}

void DTRMainWindow::onConfig(wxCommandEvent& WXUNUSED(event)) {
	DTRConfigDialog *frame2 = new DTRConfigDialog(NULL);
	frame2->Show(true);
}

void DTRMainWindow::onAbout(wxCommandEvent& WXUNUSED(event)){
	wxMessageBox(wxString::Format
	(
		"Welcome to Xinvivo's reconstruction app!\n"
		"\n"
		"This app was built for %s.",
		wxGetOsDescription()
	),
		"About Tomography Reconstruction",
		wxOK | wxICON_INFORMATION,
		this);
}

// ----------------------------------------------------------------------------
// Config frame handling
// ----------------------------------------------------------------------------

DTRConfigDialog::DTRConfigDialog(wxWindow* parent) : configDialog(parent){
	//load all values from previously saved settings
	wxConfigBase *pConfig = wxConfigBase::Get();

	estimatedDistance->SetValue(wxString::Format(wxT("%.1f"), pConfig->ReadDouble(wxT("/estimatedDistance"), 5.0f)));
	reconstructionSlices->SetValue(wxString::Format(wxT("%d"), pConfig->ReadLong(wxT("/reconstructionSlices"), 45l)));
	sliceThickness->SetValue(wxString::Format(wxT("%.1f"), pConfig->ReadDouble(wxT("/sliceThickness"), 0.5f)));
	pixelWidth->SetValue(wxString::Format(wxT("%d"), pConfig->ReadLong(wxT("/pixelWidth"), 1915l)));
	pixelHeight->SetValue(wxString::Format(wxT("%d"), pConfig->ReadLong(wxT("/pixelHeight"), 1440l)));
	pitchHeight->SetValue(wxString::Format(wxT("%.4f"), pConfig->ReadDouble(wxT("/pitchHeight"), 0.0185f)));
	pitchWidth->SetValue(wxString::Format(wxT("%.4f"), pConfig->ReadDouble(wxT("/pitchWidth"), 0.0185f)));
	for (int i = 0; i < m_grid1->GetNumberCols(); i++)
		for (int j = 0; j < m_grid1->GetNumberRows(); j++) 
			m_grid1->SetCellValue(j, i, wxString::Format(wxT("%.4f"), pConfig->ReadDouble(wxString::Format(wxT("/beamLoc%d-%d"),j,i), 0.0f)));
}

void DTRConfigDialog::onLoad(wxCommandEvent& event) {
	wxMessageBox(wxT("TODO"),
		wxT("TODO"),
		wxICON_INFORMATION | wxOK);
}

void DTRConfigDialog::onSave(wxCommandEvent& event) {
	wxMessageBox(wxT("TODO"),
		wxT("TODO"),
		wxICON_INFORMATION | wxOK);
}

void DTRConfigDialog::onOK(wxCommandEvent& event) {
	//check each value for invalid arguments
	//save each valid value to GUI config storage
	//TODO: also save each valid value to internal program storage

	//will not save all valid values, only until a bad value is hit
	double parsedDouble;
	long parsedInt = 0;
	wxConfigBase *pConfig = wxConfigBase::Get();
	if (!estimatedDistance->GetLineText(0).ToDouble(&parsedDouble)) {
		wxMessageBox(wxT("Invalid input in text box: \"Estimated distance from detector to object\"."),
			wxT("Invlaid input"),
			wxICON_STOP | wxOK);
		return;
	}
	else pConfig->Write(wxT("/estimatedDistance"), parsedDouble);

	if (!reconstructionSlices->GetLineText(0).ToLong(&parsedInt) || parsedInt <= 0) {
		wxMessageBox(wxT("Invalid input in text box: \"Number of slices to reconstruct\".\nMust be a whole number greater than 0."),
			wxT("Invlaid input"),
			wxICON_STOP | wxOK);
		return;
	}
	else pConfig->Write(wxT("/reconstructionSlices"), parsedInt);

	if (!sliceThickness->GetLineText(0).ToDouble(&parsedDouble)) {
		wxMessageBox(wxT("Invalid input in text box: \"Thickness of reconstruciton slice\"."),
			wxT("Invlaid input"),
			wxICON_STOP | wxOK);
		return;
	}
	else pConfig->Write(wxT("/sliceThickness"), parsedDouble);

	if (!pixelWidth->GetLineText(0).ToLong(&parsedInt) || parsedInt <= 0) {
		wxMessageBox(wxT("Invalid input in text box: \"Height (pixels)\".\nMust be a whole number greater than 0."),
			wxT("Invlaid input"),
			wxICON_STOP | wxOK);
		return;
	}
	else pConfig->Write(wxT("/pixelWidth"), parsedInt);

	if (!pixelHeight->GetLineText(0).ToLong(&parsedInt) || parsedInt <= 0) {
		wxMessageBox(wxT("Invalid input in text box: \"Width (pixels)\".\nMust be a whole number greater than 0."),
			wxT("Invlaid input"),
			wxICON_STOP | wxOK);
		return;
	}
	else pConfig->Write(wxT("/pixelHeight"), parsedInt);

	if (!pitchHeight->GetLineText(0).ToDouble(&parsedDouble)) {
		wxMessageBox(wxT("Invalid input in text box: \"Pitch height\"."),
			wxT("Invlaid input"),
			wxICON_STOP | wxOK);
		return;
	}
	else pConfig->Write(wxT("/pitchHeight"), parsedDouble);

	if (!pitchWidth->GetLineText(0).ToDouble(&parsedDouble)) {
		wxMessageBox(wxT("Invalid input in text box: \"Pitch width\"."),
			wxT("Invlaid input"),
			wxICON_STOP | wxOK);
		return;
	}
	else pConfig->Write(wxT("/pitchWidth"), parsedDouble);

	for (int i = 0; i < m_grid1->GetNumberCols(); i++) {
		for (int j = 0; j < m_grid1->GetNumberRows(); j++) {
			if (!m_grid1->GetCellValue(j, i).ToDouble(&parsedDouble)) {
				wxMessageBox(wxString::Format(wxT("Invalid input in text box: \"Beam emitter locations (%d,%d)\"."),j+1,i+1),//add one for average user readability
					wxT("Invlaid input"),
					wxICON_STOP | wxOK);
				return;
			}
			else pConfig->Write(wxString::Format(wxT("/beamLoc%d-%d"), j, i), parsedDouble);
		}
	}

	//All values are valid, set done flag and return
	//TODO: set done flag
	Close(true);
}


void DTRConfigDialog::onCancel(wxCommandEvent& WXUNUSED(event)) {
	Close(true);
}