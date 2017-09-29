#include "UIMain.h"

TomoError parseFile(TomoRecon * recon, const char * gainFile, const char * mainFile) {
	TomoError returnError = Tomo_OK;

	std::string ProjPath = mainFile;

	if (!ProjPath.substr(ProjPath.length() - 3, ProjPath.length()).compare("dcm")) {
		//Read projections
		int NumViews = recon->getNumViews();
		int width, height;
		recon->getProjectionDimensions(&width, &height);
		unsigned short ** RawData = new unsigned short *[NumViews];
		unsigned short ** GainData = new unsigned short *[NumViews];
		FILE * fileptr = NULL;
		std::string ProjPath = mainFile;
		std::string GainPath = gainFile;

		for (int view = 0; view < NumViews; view++) {
			//Read and correct projections
			RawData[view] = new unsigned short[width*height];
			GainData[view] = new unsigned short[width*height];

			ProjPath = ProjPath.substr(0, ProjPath.length() - 5);
			ProjPath += std::to_string(view + 1) + ".dcm";
			GainPath = GainPath.substr(0, GainPath.length() - 5);
			GainPath += std::to_string(view) + ".dcm";

			OFLog::configure(OFLogger::INFO_LOG_LEVEL);

			DicomImage *image = new DicomImage(ProjPath.c_str());

			if (image->getStatus() == EIS_Normal) {
				unsigned short *pixelData = (unsigned short *)(image->getOutputData(16));
				if (pixelData != NULL) {
					memcpy(RawData[view], pixelData, width*height * sizeof(unsigned short));
				}
				else
					std::cerr << "Error: cannot load DICOM image (" << DicomImage::getString(image->getStatus()) << ")" << endl;
			}

			delete image;

			/*fopen_s(&fileptr, GainPath.c_str(), "rb");
			if (fileptr == NULL) return Tomo_file_err;
			fread(GainData[view], sizeof(unsigned short), width*height, fileptr);
			fclose(fileptr);*/
		}

		returnError = recon->ReadProjections(GainData, RawData);

		for (int view = 0; view < NumViews; view++) {
			//Read and correct projections
			delete[] RawData[view];
			delete[] GainData[view];
		}
		delete[] RawData;
		delete[] GainData;
	}
	else recon->ReadProjectionsFromFile(gainFile, mainFile);

	return returnError;
}

// Define a new application type, each program should derive a class from wxApp
class MyApp : public wxApp{
public:
	virtual bool OnInit() wxOVERRIDE;
};

//Thread entry point, causes onInit to run like a main function would.
wxIMPLEMENT_APP(MyApp);

//Main equivalent: the program execution starts here
bool MyApp::OnInit(){
	//set name for config files in registry
	SetVendorName(wxT("Xinvivo"));

	// call the base class initialization method, parses command line inputs
	if (!wxApp::OnInit())
		return false;

	int argc = 1;
	char* argv[1] = { (char*)wxString((wxTheApp->argv)[0]).ToUTF8().data() };
	reconGlutInit(&argc, argv);

	// create the main application window
	DTRMainWindow *frame = new DTRMainWindow(NULL);
	wxString filename = wxT("C:\\Users\\jdean\\Desktop\\Patient18\\AcquiredImage1_0.raw");
#ifdef PROFILER
	frame->Show(true);
	static unsigned s_pageAdded = 0;
	frame->m_auinotebook6->AddPage(frame->CreateNewPage(filename),
		wxString::Format
		(
			wxT("%u"),
			++s_pageAdded
		),
		true);
	frame->onContinuous();
	frame->onContinuous();
	exit(0);
#else
	//Run initialization on a dummy frame that will never display, moves interop initialization time to program startup
	struct SystemControl Sys;
	frame->genSys(&Sys);
	//GLFrame* initFrame = new GLFrame(frame->m_auinotebook6, &Sys, filename, frame->m_statusBar1);
	CudaGLCanvas* m_canvas = new CudaGLCanvas(frame, frame->m_statusBar1, &Sys);
	delete m_canvas;
	frame->Show(true);
#endif

	return true;
}

DTRMainWindow::DTRMainWindow(wxWindow* parent) : mainWindow(parent){
	wxCommandEvent dummy;
	onToolbarChoice(dummy);

	wxConfigBase *pConfig = wxConfigBase::Get();
	if (pConfig == NULL)
		return;

	//size edits
	if (pConfig->Read(wxT("/dialog/max"), 0l) == 1)
		wxTopLevelWindow::Maximize(true);
	else {
		int x = pConfig->Read(wxT("/dialog/x"), 50),
			y = pConfig->Read(wxT("/dialog/y"), 50),
			w = pConfig->Read(wxT("/dialog/w"), 350),
			h = pConfig->Read(wxT("/dialog/h"), 200);
		Move(x, y);
		SetClientSize(w, h);
	}

	//Get filepath for last opened/saved file
	gainFilepath = pConfig->Read(wxT("/gainFilepath"), wxT(""));
}

//helpers
bool DTRMainWindow::checkForConsole() {
	if (m_auinotebook6->GetCurrentPage() == m_panel10) {
		//(*m_textCtrl8) << "Currently in console, cannot run. Open a new dataset with \"new\" (ctrl + n).\n";
		return true;
	}
	return false;
}

derivative_t DTRMainWindow::getEnhance() {
	derivative_t thisDisplay;
	if (xEnhance->IsChecked()) {
		if (yEnhance->IsChecked()) {
			if (absEnhance->IsChecked()) {
				thisDisplay = mag_enhance;
			}
			else {
				thisDisplay = both_enhance;
			}
		}
		else {
			if (absEnhance->IsChecked()) {
				thisDisplay = x_mag_enhance;
			}
			else {
				thisDisplay = x_enhance;
			}
		}
	}
	else {
		if (yEnhance->IsChecked()) {
			if (absEnhance->IsChecked()) {
				thisDisplay = y_mag_enhance;
			}
			else {
				thisDisplay = y_enhance;
			}
		}
		else {
			thisDisplay = no_der;
		}
	}

	return thisDisplay;
}

// event handlers
void DTRMainWindow::onNew(wxCommandEvent& WXUNUSED(event)) {
	ReconCon* rc = new ReconCon(this);
	if (rc->canceled) return;
	rc->canceled = true;
	rc->ShowModal();
	if (rc->canceled) return;

	wxFileName file = rc->filename;
	wxArrayString dirs = file.GetDirs();
	wxString name = dirs[file.GetDirCount() - 1];

	(*m_textCtrl8) << "Opening new tab titled: \"" << name << "\"\n";

	GLFrame * currentFrame = (GLFrame*)CreateNewPage(rc->filename);
	TomoRecon* recon = currentFrame->m_canvas->recon;
	m_auinotebook6->AddPage(currentFrame, name, true);
	onContinuous();
	recon->setBoundaries(rc->startDis, rc->endDis);

	setDataDisplay(currentFrame, iterRecon);
	recon->initIterative();
	bool oldLog = recon->getLogView();
	recon->setLogView(false);
	for (int i = 0; i < 15; i++) {
		recon->iterStep();
		recon->singleFrame();
		recon->resetLight();
		currentFrame->m_canvas->paint();
	}
	recon->finalizeIter();
	recon->setLogView(oldLog);
	recon->singleFrame();
	recon->resetLight();

	currentFrame->m_canvas->paint();
}

TomoError DTRMainWindow::genSys(struct SystemControl * Sys) {
	Sys->Proj.NumViews = NumViews;

	//Define new buffers to store the x,y,z locations of the x-ray focal spot array
	Sys->Geo.EmitX = new float[NumViews];
	Sys->Geo.EmitY = new float[NumViews];
	Sys->Geo.EmitZ = new float[NumViews];

	//load all values from previously saved settings
	wxConfigBase *pConfig = wxConfigBase::Get();

	//Sys->UsrIn->Orientation = pConfig->ReadLong(wxT("/orientation"), 0l) == 0l ? 0 : 1;//TODO: reintroduce
	Sys->Proj.Flip = pConfig->ReadLong(wxT("/rotationEnabled"), 0l) == 0l ? 0 : 1;

	Sys->Geo.ZPitch = 0.5f;// pConfig->ReadDouble(wxT("/sliceThickness"), 0.5f);
	Sys->Proj.Nx = pConfig->ReadLong(wxT("/pixelWidth"), 1915l);
	Sys->Proj.Ny = pConfig->ReadLong(wxT("/pixelHeight"), 1440l);
	Sys->Proj.Pitch_x = pConfig->ReadDouble(wxT("/pitchHeight"), 0.0185f);
	Sys->Proj.Pitch_y = pConfig->ReadDouble(wxT("/pitchWidth"), 0.0185f);
	for (int j = 0; j < NUMVIEWS; j++) {
		Sys->Geo.EmitX[j] = pConfig->ReadDouble(wxString::Format(wxT("/beamLoc%d-%d"), j, 0), 0.0f);
		Sys->Geo.EmitY[j] = pConfig->ReadDouble(wxString::Format(wxT("/beamLoc%d-%d"), j, 1), 0.0f);
		Sys->Geo.EmitZ[j] = pConfig->ReadDouble(wxString::Format(wxT("/beamLoc%d-%d"), j, 2), 0.0f);
	}

	return Tomo_OK;
}

wxPanel *DTRMainWindow::CreateNewPage(wxString filename) {
	struct SystemControl Sys;
	genSys(&Sys);
	wxStreamToTextRedirector redirect(m_textCtrl8);
	return new GLFrame(m_auinotebook6, &Sys, filename, m_statusBar1);
}

void DTRMainWindow::onOpen(wxCommandEvent& event) {
	if (m_auinotebook6->GetCurrentPage() == m_panel10) {
		onNew(event);
		return;
	}

	ReconCon* rc = new ReconCon(this);
	if (rc->canceled) return;
	rc->canceled = true;
	rc->ShowModal();
	if (rc->canceled) return;

	wxFileName file = rc->filename;
	wxArrayString dirs = file.GetDirs();
	wxString name = dirs[file.GetDirCount() - 1];

	(*m_textCtrl8) << "Opening new tab titled: \"" << name << "\"\n";

	GLFrame * currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;
	parseFile(recon, gainFilepath.mb_str(), rc->filename.mb_str());
	recon->setBoundaries(rc->startDis, rc->endDis);

	setDataDisplay(currentFrame, iterRecon);
	recon->resetIterative();
	bool oldLog = recon->getLogView();
	recon->setLogView(false);
	for (int i = 0; i < 15; i++) {
		recon->iterStep();
		recon->singleFrame();
		recon->resetLight();
		currentFrame->m_canvas->paint();
	}
	recon->finalizeIter();
	recon->setLogView(oldLog);
	recon->singleFrame();
	recon->resetLight();

	currentFrame->m_canvas->paint();
}

void DTRMainWindow::onSave(wxCommandEvent& event) {
	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	wxFileDialog saveFileDialog(this, _("Select a file to save as."), "", "",
		"Dicom File (*.dcm)|*.dcm", wxFD_SAVE | wxFD_OVERWRITE_PROMPT);

	if (saveFileDialog.ShowModal() == wxID_CANCEL)
		return;

	wxStreamToTextRedirector redirect(m_textCtrl8);

	m_statusBar1->SetStatusText(_("Saving data as DICOM..."));
	//recon->SaveDataAsDICOM(saveFileDialog.GetPath().ToStdString());

	std::string ProjPath = currentFrame->filename;
	int NumViews = recon->getNumViews();
	int width, height;
	recon->getProjectionDimensions(&width, &height);

	char uid[100];
	DcmFileFormat fileformat;
	DcmDataset *dataset = fileformat.getDataset();
	dataset->putAndInsertString(DCM_SOPClassUID, UID_CTImageStorage);
	dataset->putAndInsertString(DCM_SOPInstanceUID, dcmGenerateUniqueIdentifier(uid, SITE_INSTANCE_UID_ROOT));
	dataset->putAndInsertString(DCM_PatientName, "Doe^John");
	dataset->putAndInsertString(DCM_NumberOfFrames, std::to_string(NumViews).c_str());
	dataset->putAndInsertString(DCM_Rows, std::to_string(height).c_str());
	dataset->putAndInsertString(DCM_Columns, std::to_string(width).c_str());
	unsigned short * RawData = new unsigned short[width*height*NumViews];

	if (!ProjPath.substr(ProjPath.length() - 3, ProjPath.length()).compare("dcm")) {
		//Read projections
		for (int view = 0; view < NumViews; view++) {
		//int view = 0; {
			//Read projections
			ProjPath = ProjPath.substr(0, ProjPath.length() - 5);
			ProjPath += std::to_string(view) + ".dcm";//+1

			OFLog::configure(OFLogger::INFO_LOG_LEVEL);

			DicomImage *image = new DicomImage(ProjPath.c_str());

			if (image->getStatus() == EIS_Normal) {
				unsigned short *pixelData = (unsigned short *)(image->getOutputData(16));
				if (pixelData != NULL) {
					memcpy(RawData + view*width*height, pixelData, width*height * sizeof(unsigned short));
				}
				else
					std::cout << "Error: cannot load DICOM image (" << DicomImage::getString(image->getStatus()) << ")" << endl;
			}

			delete image;
		}
	}
	/*else {
		//Read projections
		unsigned short ** RawData = new unsigned short *[NumViews];
		unsigned short ** GainData = new unsigned short *[NumViews];
		FILE * fileptr = NULL;

		for (int view = 0; view < NumViews; view++) {
			//Read and correct projections
			RawData[view] = new unsigned short[Sys.Proj.Nx*Sys.Proj.Ny];
			GainData[view] = new unsigned short[Sys.Proj.Nx*Sys.Proj.Ny];

			ProjPath = ProjPath.substr(0, ProjPath.length() - 5);
			ProjPath += std::to_string(view) + ".raw";
			GainPath = GainPath.substr(0, GainPath.length() - 5);
			GainPath += std::to_string(view) + ".raw";

			fopen_s(&fileptr, ProjPath.c_str(), "rb");
			if (fileptr == NULL) return;
			fread(RawData[view], sizeof(unsigned short), Sys.Proj.Nx * Sys.Proj.Ny, fileptr);
			fclose(fileptr);

			fopen_s(&fileptr, GainPath.c_str(), "rb");
			if (fileptr == NULL) return;
			fread(GainData[view], sizeof(unsigned short), Sys.Proj.Nx * Sys.Proj.Ny, fileptr);
			fclose(fileptr);
		}

		for (int view = 0; view < NumViews; view++) {
			//Read and correct projections
			delete[] RawData[view];
			delete[] GainData[view];
		}
		delete[] RawData;
		delete[] GainData;
	}*/

	dataset->putAndInsertUint16Array(DCM_PixelData, RawData, width*height*NumViews);
	OFCondition status = fileformat.saveFile(saveFileDialog.GetPath().ToStdString(), EXS_LittleEndianExplicit);
	if (status.bad())
		std::cout << "Error: cannot write DICOM file (" << status.text() << ")" << endl;

	delete[] RawData;

	m_statusBar1->SetStatusText(_("DICOM saved!"));
}

void DTRMainWindow::onQuit(wxCommandEvent& WXUNUSED(event)){
	// true is to force the frame to close
	Close();
}

void DTRMainWindow::onResetFocus(wxCommandEvent& WXUNUSED(event)) {
	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	recon->resetFocus();
	recon->resetLight();

	currentFrame->m_canvas->paint();
}

void DTRMainWindow::onContinuous() {
	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;
	int statusWidths[] = { -4, -1, -1 };

	wxStreamToTextRedirector redirect(m_textCtrl8);
	m_statusBar1->SetFieldsCount(3, statusWidths);

	parseFile(recon, gainFilepath.mb_str(), currentFrame->filename.mb_str());

	recon->resetFocus();
	recon->resetLight();

	//Initialize all text fields for canvas
	currentFrame->m_canvas->paint(false, distanceValue, zoomSlider, zoomVal, windowSlider, windowVal, levelSlider, levelVal);
}

void DTRMainWindow::onConfig(wxCommandEvent& WXUNUSED(event)) {
	if (cfgDialog == NULL) {
		cfgDialog = new DTRConfigDialog(this);
		cfgDialog->Show(true);
	}
}

void DTRMainWindow::onGainSelect(wxCommandEvent& WXUNUSED(event)) {
	//Open files with raw extensions
	char temp[MAX_PATH];
	strncpy(temp, (const char*)gainFilepath.mb_str(), MAX_PATH - 1);
	OPENFILENAME ofn;
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = GetHWND();
	ofn.lpstrFilter = (LPCWSTR)"Raw files\0*.raw\0All Files\0*.*\0";
	ofn.lpstrFile = (LPWSTR)temp;
	ofn.nMaxFile = MAX_PATH;
	ofn.lpstrTitle = (LPCWSTR)"Select a gain file";
	ofn.Flags = OFN_DONTADDTORECENT | OFN_FILEMUSTEXIST;

	GetOpenFileNameA((LPOPENFILENAMEA)&ofn);

	gainFilepath = wxString::FromUTF8(temp);

	//Save filepath for next session
	wxConfigBase::Get()->Write(wxT("/gainFilepath"), gainFilepath);
}

void DTRMainWindow::onReconSetup(wxCommandEvent& event) {
	ReconCon* rc = new ReconCon(this);
	if (rc->canceled) return;
	rc->canceled = true;
	rc->ShowModal();
	if (rc->canceled) return;

	wxFileName file = rc->filename;
	wxArrayString dirs = file.GetDirs();
	wxString name = dirs[file.GetDirCount() - 1];

	(*m_textCtrl8) << "Opening new tab titled: \"" << name << "\"\n";

	GLFrame * currentFrame = (GLFrame*)CreateNewPage(rc->filename);
	TomoRecon* recon = currentFrame->m_canvas->recon;
	m_auinotebook6->AddPage(currentFrame, name, true);
	onContinuous();
	recon->setBoundaries(rc->startDis, rc->endDis);

	setDataDisplay(currentFrame, iterRecon);
	recon->initIterative();
	bool oldLog = recon->getLogView();
	recon->setLogView(false);
	for (int i = 0; i < 100; i++) {
		recon->iterStep();
		recon->singleFrame();
		recon->resetLight();
		currentFrame->m_canvas->paint();
	}
	recon->finalizeIter();
	recon->setLogView(oldLog);
	recon->singleFrame();
	recon->resetLight();

	currentFrame->m_canvas->paint();
}

void DTRMainWindow::onResList(wxCommandEvent& event) {
	DTRResDialog* resDialog = new DTRResDialog(this);
	resDialog->ShowModal();
	delete resDialog;
}

void DTRMainWindow::onContList(wxCommandEvent& event) {
	wxMessageBox(wxT("TODO"),
		wxT("TODO"),
		wxICON_INFORMATION | wxOK);
}

void DTRMainWindow::onRunTest(wxCommandEvent& event) {
	wxMessageBox(wxT("TODO"),
		wxT("TODO"),
		wxICON_INFORMATION | wxOK);
}

void DTRMainWindow::onTestGeo(wxCommandEvent& event) {
	wxConfigBase *pConfig = wxConfigBase::Get();
	std::vector<float> offsets = { 0.1f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 5.0f };
	std::vector<toleranceData> data;
	for (int i = 0; i < pConfig->Read(wxT("/resPhanItems"), 0l); i++){
		wxFileName filename = pConfig->Read(wxString::Format(wxT("/resPhanFile%d"), i));
		if (i == 0) {
			m_auinotebook6->AddPage(CreateNewPage(filename.GetFullPath()), wxString::Format(wxT("Geo Test %u"), i), true);
			onContinuous();
		}
		GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
		TomoRecon* recon = currentFrame->m_canvas->recon;

		if (i == 0) {
			//recon->continuousMode = true;
			recon->setDisplay(no_der);
			recon->enableNoiseMaxFilter(false);
			recon->enableScanVert(false);
			recon->enableScanHor(false);
			recon->setDataDisplay(reconstruction);
			recon->setLogView(false);
			recon->setHorFlip(false);
			recon->setVertFlip(true);
			recon->setShowNegative(true);
		}
		recon->ReadProjectionsFromFile(gainFilepath.mb_str(), filename.GetFullPath().mb_str());
		recon->singleFrame();
		recon->resetLight();

		if (data.empty()) recon->initTolerances(data, 1, offsets);

		recon->setSelBoxProj(pConfig->Read(wxString::Format(wxT("/resPhanBoxLxF%d"), i), 0l), pConfig->Read(wxString::Format(wxT("/resPhanBoxUxF%d"), i), 0l), 
			pConfig->Read(wxString::Format(wxT("/resPhanBoxLyF%d"), i), 0l), pConfig->Read(wxString::Format(wxT("/resPhanBoxUyF%d"), i), 0l));
		recon->setUpperTickProj(pConfig->Read(wxString::Format(wxT("/resPhanUpx%d"), i), 0l), pConfig->Read(wxString::Format(wxT("/resPhanUpy%d"), i), 0l));
		recon->setLowerTickProj(pConfig->Read(wxString::Format(wxT("/resPhanLowx%d"), i), 0l), pConfig->Read(wxString::Format(wxT("/resPhanLowy%d"), i), 0l));
		recon->setInputVeritcal(pConfig->Read(wxString::Format(wxT("/resPhanVert%d"), i), 0l) == 1);

		recon->setReconBox(0);
		recon->autoFocus(true);
		currentFrame->m_canvas->paint();
		recon->setReconBox(0);
		while (recon->autoFocus(false) == Tomo_OK) {
			recon->setReconBox(0);
			currentFrame->m_canvas->paint();
		}

		//switch from autofocus box to area of interest
		recon->setSelBoxProj(pConfig->Read(wxString::Format(wxT("/resPhanBoxLx%d"), i), 0l), pConfig->Read(wxString::Format(wxT("/resPhanBoxUx%d"), i), 0l),
			pConfig->Read(wxString::Format(wxT("/resPhanBoxLy%d"), i), 0l), pConfig->Read(wxString::Format(wxT("/resPhanBoxUy%d"), i), 0l));
		recon->setReconBox(0);

		recon->singleFrame();
		recon->autoLight();
		currentFrame->m_canvas->paint();

		int output = 0;
		std::ofstream FILE;
		m_statusBar1->SetStatusText(filename.GetFullPath());
		FILE.open(wxString::Format(wxT("%s\\testResults.txt"), filename.GetPath()).mb_str());
		recon->testTolerances(data, true);
		currentFrame->m_canvas->paint();
		while (recon->testTolerances(data, false) == Tomo_OK) {
			currentFrame->m_canvas->paint();
			FILE << data[output].name << ", " << data[output].numViewsChanged << ", " << data[output].viewsChanged << ", " 
				<< data[output].offset << ", " << data[output].thisDir << ", " << data[output].phantomData << "\n";
			output++;
		}
		FILE.close();

		recon->autoLight();
		currentFrame->m_canvas->paint();
	}
}

void DTRMainWindow::onAutoGeo(wxCommandEvent& event) {
	wxMessageBox(wxT("TODO"),
		wxT("TODO"),
		wxICON_INFORMATION | wxOK);
}

void DTRMainWindow::onAbout(wxCommandEvent& WXUNUSED(event)){
	wxMessageBox(wxString::Format(
		"Welcome to Xinvivo's reconstruction app!\n"
		"\n"
		"This app was built for %s.",
		wxGetOsDescription()),
		"About Tomography Reconstruction",
		wxOK | wxICON_INFORMATION,
		this);
}

void DTRMainWindow::onPageChange(wxAuiNotebookEvent& event) {
	event.Skip();//Required to actually switch the tab
	int temp = event.GetSelection();
	if (temp == 0) {//console selected
		//disable close button and all options that are not applied on new window open
		m_auinotebook6->SetWindowStyle(NULL);
		m_menubar1->Enable(m_menubar1->FindMenuItem(_("File"), _("Export")), false);
		distanceValue->Enable(false);
		autoFocus->Enable(false);
		autoLight->Enable(false);
		windowSlider->Enable(false);
		levelSlider->Enable(false);
		zoomSlider->Enable(false);
		autoAll->Enable(false);
		return;
	}
	//re-enable all controls
	m_auinotebook6->SetWindowStyle(wxAUI_NB_DEFAULT_STYLE);
	m_menubar1->Enable(m_menubar1->FindMenuItem(_("File"), _("Export")), true);
	distanceValue->Enable();
	autoFocus->Enable();
	autoLight->Enable();
	windowSlider->Enable();
	levelSlider->Enable();
	zoomSlider->Enable();
	autoAll->Enable();

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetPage(temp);
	TomoRecon* recon = currentFrame->m_canvas->recon;

	//set all control values when switching tabs
	//Distance
	distanceValue->SetValue(wxString::Format(wxT("%.2f"), recon->getDistance()));

	//Step size
	float val = recon->getStep();
	stepVal->SetLabelText(wxString::Format(wxT("%1.1f"), val));
	stepSlider->SetValue((int)(val * STEPFACTOR));

	//Window and level
	unsigned int minVal, maxVal;
	recon->getLight(&minVal, &maxVal);
	windowVal->SetLabelText(wxString::Format(wxT("%d"), maxVal));
	levelVal->SetLabelText(wxString::Format(wxT("%d"), minVal));
	windowSlider->SetValue(maxVal / WINLVLFACTOR);
	levelSlider->SetValue(minVal / WINLVLFACTOR);

	//Zoom
	int zoom = recon->getZoom();
	zoomVal->SetLabelText(wxString::Format(wxT("%5.2f"), pow(ZOOMFACTOR, zoom)));
	zoomSlider->SetValue(zoom);

	//Checkboxes
	vertFlip->SetValue(recon->getVertFlip());
	horFlip->SetValue(recon->getHorFlip());
	logView->SetValue(recon->getLogView());
	setDataDisplay(currentFrame, recon->getDataDisplay());

	//Edge enhancement
	derivative_t display = recon->getDisplay();
	xEnhance->SetValue(display == mag_enhance || display == both_enhance || display == x_enhance || display == x_mag_enhance);
	yEnhance->SetValue(display == mag_enhance || display == both_enhance || display == y_enhance || display == y_mag_enhance);
	absEnhance->SetValue(display == mag_enhance || display == x_mag_enhance || display == y_mag_enhance);
	float ratio = recon->getEnhanceRatio();
	ratioValue->SetLabelText(wxString::Format(wxT("%2.1f"), ratio));
	enhanceSlider->SetValue(ratio * ENHANCEFACTOR);

	//Scan line removal
	if (recon->scanVertIsEnabled()) {
		scanVertValue->Enable(true);
		resetScanVert->Enable(true);
		scanVertSlider->Enable(true);
	}
	else {
		scanVertValue->Enable(false);
		resetScanVert->Enable(false);
		scanVertSlider->Enable(false);
	}
	scanVertEnable->SetValue(recon->scanVertIsEnabled());
	if (recon->scanHorIsEnabled()) {
		scanHorValue->Enable(true);
		resetScanHor->Enable(true);
		scanHorSlider->Enable(true);
	}
	else {
		scanHorValue->Enable(false);
		resetScanHor->Enable(false);
		scanHorSlider->Enable(false);
	}
	scanHorEnable->SetValue(recon->scanHorIsEnabled());
	float vert = recon->getScanVertVal();
	scanVertValue->SetLabelText(wxString::Format(wxT("%1.2f"), vert));
	scanVertSlider->SetValue((int)(vert * SCANFACTOR));
	float hor = recon->getScanHorVal();
	scanHorValue->SetLabelText(wxString::Format(wxT("%1.2f"), hor));
	scanHorSlider->SetValue((int)(hor * SCANFACTOR));

	//Outlier denoising
	if (recon->noiseMaxFilterIsEnabled()) {
		noiseMaxVal->Enable(true);
		resetNoiseMax->Enable(true);
		noiseMaxSlider->Enable(true);
	}
	else {
		noiseMaxVal->Enable(false);
		resetNoiseMax->Enable(false);
		noiseMaxSlider->Enable(false);
	}
	outlierEnable->SetValue(recon->noiseMaxFilterIsEnabled());
	int max = recon->getNoiseMaxVal();
	noiseMaxVal->SetLabelText(wxString::Format(wxT("%d"), max));
	noiseMaxSlider->SetValue(max);

	//TV Denoising
	if (recon->TVIsEnabled()) {
		lambdaVal->Enable(true);
		resetLambda->Enable(true);
		lambdaSlider->Enable(true);
		iterLabel->Enable(true);
		iterVal->Enable(true);
		resetIter->Enable(true);
		iterSlider->Enable(true);
	}
	else {
		lambdaVal->Enable(false);
		resetLambda->Enable(false);
		lambdaSlider->Enable(false);
		iterLabel->Enable(false);
		iterVal->Enable(false);
		resetIter->Enable(false);
		iterSlider->Enable(false);
	}
	TVEnable->SetValue(recon->TVIsEnabled());
	int lambda = recon->getTVLambda();
	lambdaVal->SetLabelText(wxString::Format(wxT("%d"), lambda));
	lambdaSlider->SetValue(lambda);
	int iter = recon->getTVIter();
	iterVal->SetLabelText(wxString::Format(wxT("%d"), iter));
	iterSlider->SetValue(iter);

	return;
}

void DTRMainWindow::onPageClose(wxAuiNotebookEvent& event) {
	if (checkForConsole()) event.Veto();
}

//Toolbar Functions

//Navigation
void DTRMainWindow::onDistance(wxCommandEvent& event) {
	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;
	
	double distVal;
	distanceValue->GetValue().ToCDouble(&distVal);
	recon->setDistance((float)distVal);
	recon->singleFrame();
	currentFrame->m_canvas->paint(true);
}

void DTRMainWindow::onAutoFocus(wxCommandEvent& event) {
	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	if (recon->selBoxReady()) {
		//if they're greater than 0, the box was clicked and dragged successfully
		recon->autoFocus(true);
		while (recon->autoFocus(false) == Tomo_OK);
	}
	else recon->resetFocus();
	currentFrame->m_canvas->paint();
}

void DTRMainWindow::onStepSlider(wxScrollEvent& event) {
	float value = event.GetPosition();
	stepVal->SetLabelText(wxString::Format(wxT("%1.1f"), value / STEPFACTOR));

	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	recon->setStep(value / STEPFACTOR);
	parseFile(recon, gainFilepath.mb_str(), currentFrame->filename.mb_str());
	recon->singleFrame();
	currentFrame->m_canvas->paint();
}

void DTRMainWindow::onAutoLight(wxCommandEvent& event) {
	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	if (recon->selBoxReady()) {
		//if they're greater than 0, the box was clicked and dragged successfully
		recon->autoLight();
	}
	else recon->resetLight();
	currentFrame->m_canvas->paint();
}

void DTRMainWindow::onWindowSlider(wxScrollEvent& event) {
	int value = event.GetPosition();
	windowVal->SetLabelText(wxString::Format(wxT("%d"), value * WINLVLFACTOR));

	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	recon->setLight(levelSlider->GetValue() * WINLVLFACTOR, value * WINLVLFACTOR);
	currentFrame->m_canvas->paint(true);
}

void DTRMainWindow::onLevelSlider(wxScrollEvent& event) {
	int value = event.GetPosition();
	levelVal->SetLabelText(wxString::Format(wxT("%d"), value * WINLVLFACTOR));

	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	recon->setLight(value * WINLVLFACTOR, windowSlider->GetValue() * WINLVLFACTOR);
	currentFrame->m_canvas->paint(true);
}

void DTRMainWindow::onZoomSlider(wxScrollEvent& event) {
	float value = event.GetPosition();
	zoomVal->SetLabelText(wxString::Format(wxT("%5.2f"), pow(ZOOMFACTOR, value)));

	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	recon->setZoom(value);
	currentFrame->m_canvas->paint(true);
}

void DTRMainWindow::onAutoAll(wxCommandEvent& event) {
	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	if (recon->selBoxReady()) {
		//if they're greater than 0, the box was clicked and dragged successfully
		recon->autoFocus(true);
		while (recon->autoFocus(false) == Tomo_OK);
		recon->autoLight();
	}
	else {
		recon->resetFocus();
		recon->resetLight();
	}
	currentFrame->m_canvas->paint();
}

void DTRMainWindow::onVertFlip(wxCommandEvent& event) {
	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	recon->setVertFlip(vertFlip->IsChecked());
	currentFrame->m_canvas->paint();
}

void DTRMainWindow::onHorFlip(wxCommandEvent& event) {
	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	recon->setHorFlip(horFlip->IsChecked());
	currentFrame->m_canvas->paint();
}

void DTRMainWindow::onLogView(wxCommandEvent& event) {
	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	recon->setLogView(logView->IsChecked());
	recon->singleFrame();
	recon->resetLight();
	currentFrame->m_canvas->paint();
}

void DTRMainWindow::onDataDisplay(wxCommandEvent& event) {
	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;
	
	setDataDisplay(currentFrame, (sourceData)dataDisplay->GetSelection());

	recon->singleFrame();
	recon->resetLight();
	currentFrame->m_canvas->paint();
}

void DTRMainWindow::setDataDisplay(GLFrame* currentFrame, sourceData selection) {
	static int projIndex = 0;
	static int reconIndex = 0;

	TomoRecon* recon = currentFrame->m_canvas->recon;

	dataDisplay->SetSelection(selection);

	switch (recon->getDataDisplay()) {
	case projections:
		projIndex = recon->getActiveProjection();
		break;
	case iterRecon:
		reconIndex = recon->getActiveProjection();
		break;
	default:
		break;
	}

	recon->setDataDisplay(selection);
	switch (selection) {
	case projections:
		currentFrame->showScrollBar(NUMVIEWS, projIndex);
		recon->setActiveProjection(projIndex);
		break;
	case reconstruction:
		currentFrame->hideScrollBar();
		break;
	case iterRecon:
		currentFrame->showScrollBar(recon->getNumSlices(), reconIndex);
		recon->setActiveProjection(reconIndex);
		break;
	default:
		currentFrame->hideScrollBar();
		break;
	}
}

//Edge enhancement
void DTRMainWindow::onToolbarChoice(wxCommandEvent& WXUNUSED(event)) {
	//Disable all toolbars
	navToolbar->Show(false);
	edgeToolbar->Show(false);
	scanToolbar->Show(false);
	noiseToolbar->Show(false);

	//enable toolbar by selection
	switch (optionBox->GetSelection()) {
	case 0:
		navToolbar->Show(true);
		navToolbar->Realize();
		break;
	case 1:
		edgeToolbar->Show(true);
		edgeToolbar->Realize();
		break;
	case 2:
		scanToolbar->Show(true);
		scanToolbar->Realize();
		break;
	case 3:
		noiseToolbar->Show(true);
		noiseToolbar->Realize();
		break;
	}
}

void DTRMainWindow::onXEnhance(wxCommandEvent& WXUNUSED(event)) {
	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	recon->setDisplay(getEnhance());
	recon->singleFrame();
	recon->resetLight();
	currentFrame->m_canvas->paint();
}

void DTRMainWindow::onYEnhance(wxCommandEvent& WXUNUSED(event)) {
	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	recon->setDisplay(getEnhance());
	recon->singleFrame();
	recon->resetLight();
	currentFrame->m_canvas->paint();
}

void DTRMainWindow::onAbsEnhance(wxCommandEvent& WXUNUSED(event)) {
	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	recon->setDisplay(getEnhance());
	recon->singleFrame();
	recon->resetLight();
	currentFrame->m_canvas->paint();
}

void DTRMainWindow::onResetEnhance(wxCommandEvent& WXUNUSED(event)) {
	ratioValue->SetLabelText(wxString::Format(wxT("%2.1f"), ENHANCEDEFAULT));
	enhanceSlider->SetValue((int)(ENHANCEDEFAULT * ENHANCEFACTOR));

	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	recon->setEnhanceRatio(ENHANCEDEFAULT);
	recon->singleFrame();
	recon->resetLight();
	currentFrame->m_canvas->paint();
}

void DTRMainWindow::onEnhanceRatio(wxScrollEvent& event) {
	float value = (float)event.GetPosition() / ENHANCEFACTOR;
	ratioValue->SetLabelText(wxString::Format(wxT("%2.2f"), value));

	if (checkForConsole()) return;
	
	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	recon->setEnhanceRatio(value);
	recon->singleFrame();
	recon->resetLight();
	currentFrame->m_canvas->paint();
}

//Scanline correcction
void DTRMainWindow::onScanVertEnable(wxCommandEvent& event) {
	if (scanVertEnable->IsChecked()) {
		scanVertValue->Enable(true);
		resetScanVert->Enable(true);
		scanVertSlider->Enable(true);
	}
	else {
		scanVertValue->Enable(false);
		resetScanVert->Enable(false);
		scanVertSlider->Enable(false);
	}

	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	recon->enableScanVert(scanVertEnable->IsChecked());
	parseFile(recon, gainFilepath.mb_str(), currentFrame->filename.mb_str());
	recon->singleFrame();
	currentFrame->m_canvas->paint();
}

void DTRMainWindow::onScanVert(wxScrollEvent& event) {
	float value = event.GetPosition();
	scanVertValue->SetLabelText(wxString::Format(wxT("%1.2f"), value / SCANFACTOR));

	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	recon->setScanVertVal(value / SCANFACTOR);
	parseFile(recon, gainFilepath.mb_str(), currentFrame->filename.mb_str());
	recon->singleFrame();
	currentFrame->m_canvas->paint();
}

void DTRMainWindow::onResetScanVert(wxCommandEvent& event) {
	scanVertValue->SetLabelText(wxString::Format(wxT("%1.2f"), SCANVERTDEFAULT));
	scanVertSlider->SetValue(SCANVERTDEFAULT * SCANFACTOR);

	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	recon->setScanVertVal(SCANVERTDEFAULT);
	parseFile(recon, gainFilepath.mb_str(), currentFrame->filename.mb_str());
	recon->singleFrame();
	currentFrame->m_canvas->paint();
}

void DTRMainWindow::onScanHorEnable(wxCommandEvent& event) {
	if (scanHorEnable->IsChecked()) {
		scanHorValue->Enable(true);
		resetScanHor->Enable(true);
		scanHorSlider->Enable(true);
	}
	else {
		scanHorValue->Enable(false);
		resetScanHor->Enable(false);
		scanHorSlider->Enable(false);
	}

	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	recon->enableScanHor(scanHorEnable->IsChecked());
	parseFile(recon, gainFilepath.mb_str(), currentFrame->filename.mb_str());
	recon->singleFrame();
	currentFrame->m_canvas->paint();
}

void DTRMainWindow::onScanHor(wxScrollEvent& event) {
	float value = event.GetPosition();
	scanHorValue->SetLabelText(wxString::Format(wxT("%1.2f"), value / SCANFACTOR));

	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	recon->setScanHorVal(value / SCANFACTOR);
	parseFile(recon, gainFilepath.mb_str(), currentFrame->filename.mb_str());
	recon->singleFrame();
	currentFrame->m_canvas->paint();
}

void DTRMainWindow::onResetScanHor(wxCommandEvent& event) {
	scanHorValue->SetLabelText(wxString::Format(wxT("%1.2f"), SCANHORDEFAULT));
	scanHorSlider->SetValue(SCANHORDEFAULT * SCANFACTOR);

	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	recon->setScanHorVal(SCANHORDEFAULT);
	parseFile(recon, gainFilepath.mb_str(), currentFrame->filename.mb_str());
	recon->singleFrame();
	currentFrame->m_canvas->paint();
}

//Denoising
void DTRMainWindow::onNoiseMax(wxScrollEvent& event) {
	int value = event.GetPosition();
	noiseMaxVal->SetLabelText(wxString::Format(wxT("%d"), value));

	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	recon->setNoiseMaxVal(noiseMaxSlider->GetValue());
	parseFile(recon, gainFilepath.mb_str(), currentFrame->filename.mb_str());
	recon->singleFrame();
	currentFrame->m_canvas->paint();
}

void DTRMainWindow::onResetNoiseMax(wxCommandEvent& WXUNUSED(event)) {
	noiseMaxVal->SetLabelText(wxString::Format(wxT("%d"), NOISEMAXDEFAULT));
	noiseMaxSlider->SetValue(NOISEMAXDEFAULT);

	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	recon->setNoiseMaxVal(noiseMaxSlider->GetValue());
	parseFile(recon, gainFilepath.mb_str(), currentFrame->filename.mb_str());
	recon->singleFrame();
	currentFrame->m_canvas->paint();
}

void DTRMainWindow::onNoiseMaxEnable(wxCommandEvent& WXUNUSED(event)) {
	if (outlierEnable->IsChecked()) {
		noiseMaxVal->Enable(true);
		resetNoiseMax->Enable(true);
		noiseMaxSlider->Enable(true);
	}
	else {
		noiseMaxVal->Enable(false);
		resetNoiseMax->Enable(false);
		noiseMaxSlider->Enable(false);
	}

	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	recon->enableNoiseMaxFilter(outlierEnable->IsChecked());
	parseFile(recon, gainFilepath.mb_str(), currentFrame->filename.mb_str());
	recon->singleFrame();
	currentFrame->m_canvas->paint();
}

void DTRMainWindow::onTVEnable(wxCommandEvent& WXUNUSED(event)) {
	if (TVEnable->IsChecked()) {
		lambdaVal->Enable(true);
		resetLambda->Enable(true);
		lambdaSlider->Enable(true);
		iterLabel->Enable(true);
		iterVal->Enable(true);
		resetIter->Enable(true);
		iterSlider->Enable(true);
	}
	else {
		lambdaVal->Enable(false);
		resetLambda->Enable(false);
		lambdaSlider->Enable(false);
		iterLabel->Enable(false);
		iterVal->Enable(false);
		resetIter->Enable(false);
		iterSlider->Enable(false);
	}

	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	recon->enableTV(TVEnable->IsChecked());
	parseFile(recon, gainFilepath.mb_str(), currentFrame->filename.mb_str());
	recon->singleFrame();
	currentFrame->m_canvas->paint();
}

void DTRMainWindow::onResetLambda(wxCommandEvent& WXUNUSED(event)) {
	lambdaVal->SetLabelText(wxString::Format(wxT("%d"), LAMBDADEFAULT));
	lambdaSlider->SetValue(LAMBDADEFAULT);

	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	recon->setTVLambda(lambdaSlider->GetValue());
	parseFile(recon, gainFilepath.mb_str(), currentFrame->filename.mb_str());
	recon->singleFrame();
	currentFrame->m_canvas->paint();
}

void DTRMainWindow::onLambdaSlider(wxScrollEvent& event) {
	int value = event.GetPosition();
	lambdaVal->SetLabelText(wxString::Format(wxT("%d"), value));

	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	recon->setTVLambda(value);
	parseFile(recon, gainFilepath.mb_str(), currentFrame->filename.mb_str());
	recon->singleFrame();
	currentFrame->m_canvas->paint();
}

void DTRMainWindow::onResetIter(wxCommandEvent& WXUNUSED(event)) {
	iterVal->SetLabelText(wxString::Format(wxT("%d"), ITERDEFAULT));
	iterSlider->SetValue(ITERDEFAULT);

	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	recon->setTVIter(iterSlider->GetValue());
	parseFile(recon, gainFilepath.mb_str(), currentFrame->filename.mb_str());
	recon->singleFrame();
	currentFrame->m_canvas->paint();
}

void DTRMainWindow::onIterSlider(wxScrollEvent& event) {
	int value = event.GetPosition();
	iterVal->SetLabelText(wxString::Format(wxT("%d"), value));

	if (checkForConsole()) return;

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	recon->setTVIter(value);
	parseFile(recon, gainFilepath.mb_str(), currentFrame->filename.mb_str());
	recon->singleFrame();
	currentFrame->m_canvas->paint();
}

DTRMainWindow::~DTRMainWindow() {
	wxConfigBase *pConfig = wxConfigBase::Get();
	if (pConfig == NULL)
		return;

	// save the frame position
	int x, y, w, h;
	GetClientSize(&w, &h);
	GetPosition(&x, &y);
	pConfig->Write(wxT("/dialog/x"), (long)x);
	pConfig->Write(wxT("/dialog/y"), (long)y);
	pConfig->Write(wxT("/dialog/w"), (long)w);
	pConfig->Write(wxT("/dialog/h"), (long)h);
	if (wxTopLevelWindow::IsMaximized())
		pConfig->Write(wxT("/dialog/max"), 1);
	else
		pConfig->Write(wxT("/dialog/max"), 0);

	//cuda(DeviceReset());//only reset here where we know all windows are finished
}

// ----------------------------------------------------------------------------
// Resolution phatom selector frame handling
// ----------------------------------------------------------------------------

ReconCon::ReconCon(wxWindow* parent) : reconConfig(parent) {
	wxFileDialog openFileDialog(this, _("Select one raw image file"), "", "",
		"Raw or DICOM Files (*.raw, *.dcm)|*.raw;*.dcm|Raw File (*.raw)|*.raw|DICOM File (*.dcm)|*.dcm", wxFD_OPEN | wxFD_FILE_MUST_EXIST);

	if (openFileDialog.ShowModal() == wxID_CANCEL)
		return;

	canceled = false;

	wxString fn(openFileDialog.GetPath());
	filename = fn;

	DTRMainWindow* typedParent = (DTRMainWindow*)GetParent();

	struct SystemControl Sys;
	typedParent->genSys(&Sys);
	wxStreamToTextRedirector redirect(typedParent->m_textCtrl8);
	drawPanel = new GLFrame(this, &Sys, filename);
	TomoRecon* recon = ((GLFrame*)drawPanel)->m_canvas->recon;
	recon->enableGain(false);
	parseFile(recon, filename.mb_str(), filename.mb_str());

	recon->resetFocus();
	recon->resetLight();

	((GLFrame*)drawPanel)->m_canvas->paint(false, distance);

	bSizer6->Add(drawPanel, 10, wxEXPAND | wxALL);
	bSizer6->Layout();
}

void ReconCon::onDistance(wxCommandEvent& event) {
	TomoRecon* recon = ((GLFrame*)drawPanel)->m_canvas->recon;

	double distVal;
	distance->GetValue().ToCDouble(&distVal);
	recon->setDistance((float)distVal);
	recon->singleFrame();
	((GLFrame*)drawPanel)->m_canvas->paint(true);
}

void ReconCon::onOk(wxCommandEvent& event) {
	startDistance->GetValue().ToCDouble(&startDis);
	endDistance->GetValue().ToCDouble(&endDis);
	canceled = false;
	Close();
}

void ReconCon::onCancel(wxCommandEvent& event) {
	canceled = true;
	Close();
}

void ReconCon::onSetStartDis(wxCommandEvent& event) {
	startDistance->SetValue(distance->GetValue());
}

void ReconCon::onSetEndDis(wxCommandEvent& event) {
	endDistance->SetValue(distance->GetValue());
}

void ReconCon::onClose(wxCloseEvent& event) {
	delete drawPanel;
	event.Skip();
}

ReconCon::~ReconCon() {
	
}

//Save dialog box handling
DTRSliceSave::DTRSliceSave(wxWindow* parent) : sliceDialog(parent) {
}

void DTRSliceSave::onSliceValue(wxCommandEvent& WXUNUSED(event)) {
	sliceValue->GetLineText(0).ToLong(&value);
	Close(true);
}

DTRSliceSave::~DTRSliceSave() {

}

// ----------------------------------------------------------------------------
// Config frame handling
// ----------------------------------------------------------------------------

DTRConfigDialog::DTRConfigDialog(wxWindow* parent) : configDialog(parent){
	//load all values from previously saved settings
	wxConfigBase *pConfig = wxConfigBase::Get();

	pixelWidth->SetValue(wxString::Format(wxT("%d"), pConfig->ReadLong(wxT("/pixelWidth"), 1915l)));
	pixelHeight->SetValue(wxString::Format(wxT("%d"), pConfig->ReadLong(wxT("/pixelHeight"), 1440l)));
	pitchHeight->SetValue(wxString::Format(wxT("%.4f"), pConfig->ReadDouble(wxT("/pitchHeight"), 0.0185f)));
	pitchWidth->SetValue(wxString::Format(wxT("%.4f"), pConfig->ReadDouble(wxT("/pitchWidth"), 0.0185f)));
	for (int i = 0; i < m_grid1->GetNumberCols(); i++)
		for (int j = 0; j < m_grid1->GetNumberRows(); j++) 
			m_grid1->SetCellValue(j, i, wxString::Format(wxT("%.4f"), pConfig->ReadDouble(wxString::Format(wxT("/beamLoc%d-%d"),j,i), 0.0f)));

	//Get filepath for last opened/saved file
	configFilepath = std::string(pConfig->Read(wxT("/configFilePath"), "").mb_str());
}

void DTRConfigDialog::onLoad(wxCommandEvent& event) {
	//Open files with txt or json extensions
	char temp[MAX_PATH];
	strncpy(temp, configFilepath.c_str(), MAX_PATH - 1);
	OPENFILENAME ofn;
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = GetHWND();
	ofn.lpstrFilter = (LPCWSTR)"JSON file\0*.json\0Text File\0*.txt\0";
	ofn.lpstrFile = (LPWSTR)temp;
	ofn.nMaxFile = MAX_PATH;
	ofn.lpstrTitle = (LPCWSTR)"Select a geometry file";
	ofn.Flags = OFN_DONTADDTORECENT | OFN_FILEMUSTEXIST;

	GetOpenFileNameA((LPOPENFILENAMEA)&ofn);

	//Set filepath for last opened/saved file
	wxConfigBase::Get()->Write(wxT("/configFilePath"), wxString::FromUTF8(temp));
	configFilepath = temp;

	//check file type and parse accordingly
	if (configFilepath.substr(configFilepath.find_last_of(".") + 1) == "json") {
		ParseJSONFile(configFilepath);
	}
	else {
		ParseLegacyTxt(configFilepath);
	}
}

TomoError DTRConfigDialog::ParseJSONFile(std::string FilePath) {
	//Open file and parse to cJSON object
	std::ifstream ifs(FilePath.c_str());
	std::string input((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
	cJSON * root = cJSON_Parse(input.c_str());

	//Populate form using parsed values
	//TODO: shittons of error checking
	pixelWidth->SetValue(wxString::Format(wxT("%d"), cJSON_GetObjectItem(root, "pixelWidth")->valueint));
	pixelHeight->SetValue(wxString::Format(wxT("%d"), cJSON_GetObjectItem(root, "pixelHeight")->valueint));
	pitchHeight->SetValue(wxString::Format(wxT("%.4f"), cJSON_GetObjectItem(root, "pitchHeight")->valuedouble));
	pitchWidth->SetValue(wxString::Format(wxT("%.4f"), cJSON_GetObjectItem(root, "pitchWidth")->valuedouble));
	for (int i = 0; i < m_grid1->GetNumberCols(); i++)
		for (int j = 0; j < m_grid1->GetNumberRows(); j++)
			m_grid1->SetCellValue(j, i, wxString::Format(wxT("%.4f"), 
				cJSON_GetObjectItem(root, (const char*)wxString::Format(wxT("beamGeo%d-%d"), j, i).mb_str(wxConvUTF8))->valuedouble));

	cJSON_Delete(root);
	return Tomo_OK;
}

TomoError DTRConfigDialog::ParseLegacyTxt(std::string FilePath) {
	//Open fstream to text file
	std::ifstream file(FilePath.c_str());

	if (!file.is_open()) {
		std::cout << "Error opening file: " << FilePath.c_str() << std::endl;
		std::cout << "Please check and re-run program." << std::endl;
		return Tomo_file_err;
	}

	//Define two character arrays to read values
	char data[1024], data_in[12];

	//skip table headers
	file.getline(data, 1024);
	bool useview = false;
	int count = 0, num = 0;

	//Cycle through the views and read geometry
	for (int view = 0; view < NUMVIEWS; view++){
		file.getline(data, 1024);//Read data line

		//Skip first coloumn: Beam Number	
		do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);
		for (int i = 0; i < 12; i++) data_in[i] = '\0';
		count++; num = 0;

		//Read second colomn: emitter x location
		do { data_in[num] = data[count];	count++; num++; } while (data[count] != '\t' && num < 12);
		m_grid1->SetCellValue(view, 0, wxString::Format(wxT("%.4f"), atof(data_in)));
		for (int i = 0; i < 12; i++) data_in[i] = '\0';
		count++; num = 0;

		//Read third colomn: emitter y location
		do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);
		m_grid1->SetCellValue(view, 1, wxString::Format(wxT("%.4f"), atof(data_in)));
		for (int i = 0; i < 12; i++) data_in[i] = '\0';
		count++; num = 0;

		//Read fourth colomn: emitter z location
		do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);
		m_grid1->SetCellValue(view, 2, wxString::Format(wxT("%.4f"), atof(data_in)));
		for (int i = 0; i < 12; i++) data_in[i] = '\0';
		count = 0; num = 0;
	}

	//Skip the next 2 lines and read the third to get estimated center of object
	file.getline(data, 1024); file.getline(data, 1024); file.getline(data, 1024);
	do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);

	//skip the next 2 lines and read third to get slice thickness
	for (int i = 0; i < 12; i++) data_in[i] = '\0';
	count = 0; num = 0;
	file.getline(data, 1024); file.getline(data, 1024); file.getline(data, 1024);
	do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);
	for (int i = 0; i < 12; i++) data_in[i] = '\0';
	count = 0; num = 0;

	//Skip the next two lines and read the third
	file.getline(data, 1024); file.getline(data, 1024); file.getline(data, 1024);

	//Read four values defining the detector size
	do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);
	pixelWidth->SetValue(wxString::Format(wxT("%d"), atoi(data_in)));
	for (int i = 0; i < 12; i++) data_in[i] = '\0';
	count++; num = 0;
	do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);
	pixelHeight->SetValue(wxString::Format(wxT("%d"), atoi(data_in)));
	for (int i = 0; i < 12; i++) data_in[i] = '\0';
	count++; num = 0;
	do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);
	pitchWidth->SetValue(wxString::Format(wxT("%.4f"), atof(data_in)));
	for (int i = 0; i < 12; i++) data_in[i] = '\0';
	count++; num = 0;
	do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);
	pitchHeight->SetValue(wxString::Format(wxT("%.4f"), atof(data_in)));
	for (int i = 0; i < 12; i++) data_in[i] = '\0';
	count = 0; num = 0;

	//Skip the next two lines and read the third to read number of slices to reconstruct
	file.getline(data, 1024); file.getline(data, 1024); file.getline(data, 1024);
	do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);
	for (int i = 0; i < 12; i++) data_in[i] = '\0';
	count = 0; num = 0;

	//Skip the next two lines and read the third to see direction of data
	file.getline(data, 1024); file.getline(data, 1024); file.getline(data, 1024);
	do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);
	for (int i = 0; i < 12; i++) data_in[i] = '\0';
	count = 0; num = 0;

	//Skip the next two lines and read the third to see if automatic offset calculation
	file.getline(data, 1024); file.getline(data, 1024); file.getline(data, 1024);
	do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);
	for (int i = 0; i < 12; i++) data_in[i] = '\0';
	count = 0; num = 0;

	// Skip the next two lines and read the third to see if automatic offset calculation
	file.getline(data, 1024); file.getline(data, 1024); file.getline(data, 1024);
	do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);
	for (int i = 0; i < 12; i++) data_in[i] = '\0';
	count = 0; num = 0;

	// Skip the next two lines and read the third to see if use TV reconstruction
	file.getline(data, 1024); file.getline(data, 1024); file.getline(data, 1024);
	do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);
	for (int i = 0; i < 12; i++) data_in[i] = '\0';
	count = 0; num = 0;

	// Skip the next two lines and read the third to check orientation
	file.getline(data, 1024); file.getline(data, 1024); file.getline(data, 1024);
	do { data_in[num] = data[count]; count++; num++; } while (data[count] != '\t' && num < 12);
	for (int i = 0; i < 12; i++) data_in[i] = '\0';
	count = 0; num = 0;

	file.close();
}

void DTRConfigDialog::onSave(wxCommandEvent& event) {
	if (checkInputs() != Tomo_OK) return;

	//Save files with json extension only
	char temp[MAX_PATH];
	strncpy(temp, configFilepath.c_str(), MAX_PATH - 1);
	OPENFILENAME ofn;
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = GetHWND();
	ofn.lpstrFilter = (LPCWSTR)"JSON file\0*.json\0";
	ofn.lpstrFile = (LPWSTR)temp;
	ofn.nMaxFile = MAX_PATH;
	ofn.lpstrTitle = (LPCWSTR)"Select a geometry file";
	ofn.Flags = OFN_DONTADDTORECENT;

	GetSaveFileNameA((LPOPENFILENAMEA)&ofn);

	//TODO: check JSON extension

	//Set filepath for last opened/saved file
	wxConfigBase::Get()->Write(wxT("/configFilePath"), wxString::FromUTF8(temp));
	configFilepath = temp;

	//Create and populate a cJSON object
	double parsedDouble;
	long parsedInt = 0;
	cJSON *root = cJSON_CreateObject();

	pitchHeight->GetLineText(0).ToDouble(&parsedDouble);
	cJSON_AddNumberToObject(root, "pitchHeight", parsedDouble);
	pitchWidth->GetLineText(0).ToDouble(&parsedDouble);
	cJSON_AddNumberToObject(root, "pitchWidth", parsedDouble);
	pixelWidth->GetLineText(0).ToLong(&parsedInt);
	cJSON_AddNumberToObject(root, "pixelWidth", parsedInt);
	pixelHeight->GetLineText(0).ToLong(&parsedInt);
	cJSON_AddNumberToObject(root, "pixelHeight", parsedInt);
	for (int i = 0; i < m_grid1->GetNumberCols(); i++)
		for (int j = 0; j < m_grid1->GetNumberRows(); j++) {
			m_grid1->GetCellValue(j, i).ToDouble(&parsedDouble);
			cJSON_AddNumberToObject(root, (const char*)wxString::Format(wxT("beamGeo%d-%d"), j, i).mb_str(wxConvUTF8), parsedDouble);
		}

	//convert to actual string
	char *rendered = cJSON_Print(root);

	//output to disk
	std::ofstream FILE;
	FILE.open(configFilepath, std::ios::binary);
	FILE << rendered;
	FILE.close();

	//cleanup structure
	cJSON_Delete(root);
}

void DTRConfigDialog::onOK(wxCommandEvent& event) {
	if(checkInputs() != Tomo_OK) return;

	//All values are valid, set done flag and return
	//TODO: set done flag
	((DTRMainWindow*)GetParent())->cfgDialog = NULL;
	Close(true);
}

TomoError DTRConfigDialog::checkInputs() {
	//check each value for invalid arguments
	//save each valid value to GUI config storage

	//will not save all valid values, only until a bad value is hit
	double parsedDouble;
	long parsedInt = 0;
	wxConfigBase *pConfig = wxConfigBase::Get();

	if (!pixelWidth->GetLineText(0).ToLong(&parsedInt) || parsedInt <= 0) {
		wxMessageBox(wxT("Invalid input in text box: \"Height (pixels)\".\nMust be a whole number greater than 0."),
			wxT("Invlaid input"),
			wxICON_STOP | wxOK);
		return Tomo_input_err;
	}
	else pConfig->Write(wxT("/pixelWidth"), parsedInt);

	if (!pixelHeight->GetLineText(0).ToLong(&parsedInt) || parsedInt <= 0) {
		wxMessageBox(wxT("Invalid input in text box: \"Width (pixels)\".\nMust be a whole number greater than 0."),
			wxT("Invlaid input"),
			wxICON_STOP | wxOK);
		return Tomo_input_err;
	}
	else pConfig->Write(wxT("/pixelHeight"), parsedInt);

	if (!pitchHeight->GetLineText(0).ToDouble(&parsedDouble)) {
		wxMessageBox(wxT("Invalid input in text box: \"Pitch height\"."),
			wxT("Invlaid input"),
			wxICON_STOP | wxOK);
		return Tomo_input_err;
	}
	else pConfig->Write(wxT("/pitchHeight"), parsedDouble);

	if (!pitchWidth->GetLineText(0).ToDouble(&parsedDouble)) {
		wxMessageBox(wxT("Invalid input in text box: \"Pitch width\"."),
			wxT("Invlaid input"),
			wxICON_STOP | wxOK);
		return Tomo_input_err;
	}
	else pConfig->Write(wxT("/pitchWidth"), parsedDouble);

	for (int i = 0; i < m_grid1->GetNumberCols(); i++) {
		for (int j = 0; j < m_grid1->GetNumberRows(); j++) {
			if (!m_grid1->GetCellValue(j, i).ToDouble(&parsedDouble)) {
				wxMessageBox(wxString::Format(wxT("Invalid input in text box: \"Beam emitter locations (%d,%d)\"."), j + 1, i + 1),//add one for average user readability
					wxT("Invlaid input"),
					wxICON_STOP | wxOK);
				return Tomo_input_err;
			}
			else pConfig->Write(wxString::Format(wxT("/beamLoc%d-%d"), j, i), parsedDouble);
		}
	}

	return Tomo_OK;
}

void DTRConfigDialog::onCancel(wxCommandEvent& WXUNUSED(event)) {
	((DTRMainWindow*)GetParent())->cfgDialog = NULL;
	Close(true);
}

void DTRConfigDialog::onConfigChar(wxCommandEvent& event) {
	wxTextCtrl* caller = (wxTextCtrl*)event.GetEventObject();
	if (caller == pixelWidth) {
		pixelHeight->SetFocus();
		pixelHeight->SelectAll();
	}
	if (caller == pixelHeight) {
		pitchHeight->SetFocus();
		pitchHeight->SelectAll();
	}
	if (caller == pitchHeight) {
		pitchWidth->SetFocus();
		pitchWidth->SelectAll();
	}
	if (caller == pitchWidth) {
		m_grid1->SetFocus();
	}
}

DTRConfigDialog::~DTRConfigDialog() {
	
}

// ----------------------------------------------------------------------------
// Resolution phatom selector frame handling
// ----------------------------------------------------------------------------

DTRResDialog::DTRResDialog(wxWindow* parent) : resDialog(parent) {
	//load all values from previously saved settings
	wxConfigBase *pConfig = wxConfigBase::Get();

	//Setup column structure    
	wxListItem col0;
	col0.SetId(0);
	col0.SetText(_("Filepaths"));
	col0.SetWidth(400);
	m_listCtrl->InsertColumn(0, col0);

	wxListItem col1;
	col1.SetId(1);
	col1.SetText(_("BoxUx"));
	col1.SetWidth(50);
	m_listCtrl->InsertColumn(1, col1);

	wxListItem col2;
	col2.SetId(2);
	col2.SetText(_("BoxUy"));
	col2.SetWidth(50);
	m_listCtrl->InsertColumn(2, col2);

	wxListItem col3;
	col3.SetId(3);
	col3.SetText(_("BoxLx"));
	col3.SetWidth(50);
	m_listCtrl->InsertColumn(3, col3);

	wxListItem col4;
	col4.SetId(4);
	col4.SetText(_("BoxLy"));
	col4.SetWidth(50);
	m_listCtrl->InsertColumn(4, col4);

	wxListItem col5;
	col5.SetId(5);
	col5.SetText(_("Lowx"));
	col5.SetWidth(50);
	m_listCtrl->InsertColumn(5, col5);

	wxListItem col6;
	col6.SetId(6);
	col6.SetText(_("Lowy"));
	col6.SetWidth(50);
	m_listCtrl->InsertColumn(6, col6);

	wxListItem col7;
	col7.SetId(7);
	col7.SetText(_("Upx"));
	col7.SetWidth(50);
	m_listCtrl->InsertColumn(7, col7);

	wxListItem col8;
	col8.SetId(8);
	col8.SetText(_("Upy"));
	col8.SetWidth(50);
	m_listCtrl->InsertColumn(8, col8);

	wxListItem col9;
	col9.SetId(9);
	col9.SetText(_("Vert?"));
	col9.SetWidth(50);
	m_listCtrl->InsertColumn(9, col9);

	wxListItem col10;
	col10.SetId(10);
	col10.SetText(_("BoxUxF"));
	col10.SetWidth(50);
	m_listCtrl->InsertColumn(10, col10);

	wxListItem col11;
	col11.SetId(11);
	col11.SetText(_("BoxUyF"));
	col11.SetWidth(50);
	m_listCtrl->InsertColumn(11, col11);

	wxListItem col12;
	col12.SetId(12);
	col12.SetText(_("BoxLxF"));
	col12.SetWidth(50);
	m_listCtrl->InsertColumn(12, col12);

	wxListItem col13;
	col13.SetId(13);
	col13.SetText(_("BoxLyF"));
	col13.SetWidth(50);
	m_listCtrl->InsertColumn(13, col13);

	for (int i = 0; i < pConfig->Read(wxT("/resPhanItems"), 0l); i++) {
		m_listCtrl->InsertItem(i, pConfig->Read(wxString::Format(wxT("/resPhanFile%d"), i), wxT("")));
		m_listCtrl->SetItem(i, 1, wxString::Format(wxT("%d"),pConfig->Read(wxString::Format(wxT("/resPhanBoxUx%d"), i), 0l)));
		m_listCtrl->SetItem(i, 2, wxString::Format(wxT("%d"), pConfig->Read(wxString::Format(wxT("/resPhanBoxUy%d"), i), 0l)));
		m_listCtrl->SetItem(i, 3, wxString::Format(wxT("%d"), pConfig->Read(wxString::Format(wxT("/resPhanBoxLx%d"), i), 0l)));
		m_listCtrl->SetItem(i, 4, wxString::Format(wxT("%d"), pConfig->Read(wxString::Format(wxT("/resPhanBoxLy%d"), i), 0l)));
		m_listCtrl->SetItem(i, 5, wxString::Format(wxT("%d"), pConfig->Read(wxString::Format(wxT("/resPhanLowx%d"), i), 0l)));
		m_listCtrl->SetItem(i, 6, wxString::Format(wxT("%d"), pConfig->Read(wxString::Format(wxT("/resPhanLowy%d"), i), 0l)));
		m_listCtrl->SetItem(i, 7, wxString::Format(wxT("%d"), pConfig->Read(wxString::Format(wxT("/resPhanUpx%d"), i), 0l)));
		m_listCtrl->SetItem(i, 8, wxString::Format(wxT("%d"), pConfig->Read(wxString::Format(wxT("/resPhanUpy%d"), i), 0l)));
		m_listCtrl->SetItem(i, 9, pConfig->Read(wxString::Format(wxT("/resPhanVert%d"), i), 0l) == 1 ? wxT("Yes") : wxT("No"));
		m_listCtrl->SetItem(i, 10, wxString::Format(wxT("%d"), pConfig->Read(wxString::Format(wxT("/resPhanBoxUxF%d"), i), 0l)));
		m_listCtrl->SetItem(i, 11, wxString::Format(wxT("%d"), pConfig->Read(wxString::Format(wxT("/resPhanBoxUyF%d"), i), 0l)));
		m_listCtrl->SetItem(i, 12, wxString::Format(wxT("%d"), pConfig->Read(wxString::Format(wxT("/resPhanBoxLxF%d"), i), 0l)));
		m_listCtrl->SetItem(i, 13, wxString::Format(wxT("%d"), pConfig->Read(wxString::Format(wxT("/resPhanBoxLyF%d"), i), 0l)));
	}
}

void DTRResDialog::onAddNew(wxCommandEvent& event) {
	wxFileDialog openFileDialog(this, _("Select one raw image file"), "", "",
		"Raw or DICOM Files (*.raw, *.dcm)|*.raw;*.dcm|Raw File (*.raw)|*.raw|DICOM File (*.dcm)|*.dcm", wxFD_OPEN | wxFD_FILE_MUST_EXIST);

	if (openFileDialog.ShowModal() == wxID_CANCEL)
		return;

	int vertical = wxMessageBox(wxT("Is the selected input vertical? (no=horizontal)"), 
		wxT("Input orientation"),
		wxICON_INFORMATION | wxYES | wxNO);

	struct SystemControl Sys;
	((DTRMainWindow*)GetParent())->genSys(&Sys);
	frame = new GLWindow(this, vertical == wxYES, &Sys, ((DTRMainWindow*)GetParent())->gainFilepath, openFileDialog.GetPath());
	
	{
		TomoRecon* recon = frame->m_canvas->recon;

		recon->setDisplay(no_der);
		recon->enableNoiseMaxFilter(false);
		recon->enableScanVert(false);
		recon->enableScanHor(false);
		recon->setDataDisplay(projections);
		recon->setLogView(true);
		recon->setHorFlip(false);
		recon->setVertFlip(true);
		recon->enableTV(false);

		recon->ReadProjectionsFromFile(((DTRMainWindow*)GetParent())->gainFilepath.mb_str(), openFileDialog.GetPath().mb_str());
		recon->singleFrame();

		recon->resetLight();
	}

	int res = frame->ShowModal();

	if (res == wxID_OK) {
		//User successfully completed the dialog interaction
		wxString file = openFileDialog.GetPath();
		int index = m_listCtrl->FindItem(-1, file);
		if (index == wxNOT_FOUND)
			index = m_listCtrl->InsertItem(0, file);

		int x1, y1, x2, y2, lowX, lowY, upX, upY;
		frame->m_canvas->recon->getSelBoxRaw(&x1, &x2, &y1, &y2);
		frame->m_canvas->recon->getUpperTickRaw(&upX, &upY);
		frame->m_canvas->recon->getLowerTickRaw(&lowX, &lowY);
		m_listCtrl->SetItem(index, 1, wxString::Format(wxT("%d"), x1));
		m_listCtrl->SetItem(index, 2, wxString::Format(wxT("%d"), y1));
		m_listCtrl->SetItem(index, 3, wxString::Format(wxT("%d"), x2));
		m_listCtrl->SetItem(index, 4, wxString::Format(wxT("%d"), y2));
		m_listCtrl->SetItem(index, 5, wxString::Format(wxT("%d"), lowX));
		m_listCtrl->SetItem(index, 6, wxString::Format(wxT("%d"), lowY));
		m_listCtrl->SetItem(index, 7, wxString::Format(wxT("%d"), upX));
		m_listCtrl->SetItem(index, 8, wxString::Format(wxT("%d"), upY));
		m_listCtrl->SetItem(index, 9, vertical == wxYES ? wxT("Yes") : wxT("No"));
		m_listCtrl->SetItem(index, 10, wxString::Format(wxT("%d"), frame->m_canvas->x1));
		m_listCtrl->SetItem(index, 11, wxString::Format(wxT("%d"), frame->m_canvas->y1));
		m_listCtrl->SetItem(index, 12, wxString::Format(wxT("%d"), frame->m_canvas->x2));
		m_listCtrl->SetItem(index, 13, wxString::Format(wxT("%d"), frame->m_canvas->y2));
	}
}

void DTRResDialog::onRemove(wxCommandEvent& event) {
	int selection = wxNOT_FOUND;
	while (true) {
		selection = m_listCtrl->GetNextItem(wxNOT_FOUND, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
		if (selection == wxNOT_FOUND) break;
		m_listCtrl->DeleteItem(selection);//stride backwards to avoid indexing issues
	}
}

void DTRResDialog::onOk(wxCommandEvent& event) {
	//save values using a saved array value
	//currently, it does no garbage collection if fewer filenames are saved
	wxConfigBase *pConfig = wxConfigBase::Get();

	//save array size
	int items = m_listCtrl->GetItemCount();
	pConfig->Write(wxT("/resPhanItems"), items);

	int selection = wxNOT_FOUND;
	while (true) {
		selection = m_listCtrl->GetNextItem(selection);
		if (selection == wxNOT_FOUND) break;

		//find the selection
		wxListItem item;
		long value;
		item.m_itemId = selection;
		item.m_mask = wxLIST_MASK_TEXT;

		//"iterate" through the columns
		pConfig->Write(wxString::Format(wxT("/resPhanFile%d"), selection), m_listCtrl->GetItemText(selection));

		item.m_col = 1;
		m_listCtrl->GetItem(item);
		item.m_text.ToLong(&value);
		pConfig->Write(wxString::Format(wxT("/resPhanBoxUx%d"), selection), value);

		item.m_col = 2;
		m_listCtrl->GetItem(item);
		item.m_text.ToLong(&value);
		pConfig->Write(wxString::Format(wxT("/resPhanBoxUy%d"), selection), value);

		item.m_col = 3;
		m_listCtrl->GetItem(item);
		item.m_text.ToLong(&value);
		pConfig->Write(wxString::Format(wxT("/resPhanBoxLx%d"), selection), value);

		item.m_col = 4;
		m_listCtrl->GetItem(item);
		item.m_text.ToLong(&value);
		pConfig->Write(wxString::Format(wxT("/resPhanBoxLy%d"), selection), value);

		item.m_col = 5;
		m_listCtrl->GetItem(item);
		item.m_text.ToLong(&value);
		pConfig->Write(wxString::Format(wxT("/resPhanLowx%d"), selection), value);

		item.m_col = 6;
		m_listCtrl->GetItem(item);
		item.m_text.ToLong(&value);
		pConfig->Write(wxString::Format(wxT("/resPhanLowy%d"), selection), value);

		item.m_col = 7;
		m_listCtrl->GetItem(item);
		item.m_text.ToLong(&value);
		pConfig->Write(wxString::Format(wxT("/resPhanUpx%d"), selection), value);

		item.m_col = 8;
		m_listCtrl->GetItem(item);
		item.m_text.ToLong(&value);
		pConfig->Write(wxString::Format(wxT("/resPhanUpy%d"), selection), value);

		item.m_col = 9;
		m_listCtrl->GetItem(item);
		if(item.m_text == wxT("Yes"))
			pConfig->Write(wxString::Format(wxT("/resPhanVert%d"), selection), 1l);
		else
			pConfig->Write(wxString::Format(wxT("/resPhanVert%d"), selection), 0l);

		item.m_col = 10;
		m_listCtrl->GetItem(item);
		item.m_text.ToLong(&value);
		pConfig->Write(wxString::Format(wxT("/resPhanBoxUxF%d"), selection), value);

		item.m_col = 11;
		m_listCtrl->GetItem(item);
		item.m_text.ToLong(&value);
		pConfig->Write(wxString::Format(wxT("/resPhanBoxUyF%d"), selection), value);

		item.m_col = 12;
		m_listCtrl->GetItem(item);
		item.m_text.ToLong(&value);
		pConfig->Write(wxString::Format(wxT("/resPhanBoxLxF%d"), selection), value);

		item.m_col = 13;
		m_listCtrl->GetItem(item);
		item.m_text.ToLong(&value);
		pConfig->Write(wxString::Format(wxT("/resPhanBoxLyF%d"), selection), value);
	}

	Close(true);
}

void DTRResDialog::onCancel(wxCommandEvent& event) {
	Close(true);
}

DTRResDialog::~DTRResDialog() {}

wxBEGIN_EVENT_TABLE(GLFrame, wxPanel)
EVT_SCROLL(GLFrame::OnScroll)
EVT_MOUSEWHEEL(GLFrame::OnMousewheel)
wxEND_EVENT_TABLE()

GLFrame::GLFrame(wxWindow *frame, struct SystemControl * Sys, wxString filename, wxStatusBar* status,
	const wxPoint& pos, const wxSize& size, long style)
	: wxPanel(frame, wxID_ANY, pos, size), m_canvas(NULL), m_status(status), filename(filename){
	//initialize the canvas to this object
	m_canvas = new CudaGLCanvas(this, status, Sys, wxID_ANY, NULL, GetClientSize());
	m_scrollBar = new wxScrollBar(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSB_HORIZONTAL);
	bSizer = new wxBoxSizer(wxVERTICAL);

	bSizer->Add(m_canvas, 1, wxEXPAND | wxALL, 5);
	m_scrollBar->SetScrollbar(0, 1, NUMVIEWS, 1);
	bSizer->Add(m_scrollBar, 0, wxALL | wxEXPAND, 5);

	m_scrollBar->Show(true);

	this->SetSizer(bSizer);
	bSizer->Fit(this);

	hideScrollBar();

	// Show the frame
	Show(true);
	Raise();//grab attention when the frame has finished rendering
}

GLFrame::~GLFrame(){
	delete m_canvas;
}

void GLFrame::hideScrollBar() {
	bSizer->Hide(1);
	this->Layout();
}

void GLFrame::showScrollBar(int steps, int current) {
	m_scrollBar->SetScrollbar(current, 1, steps, 1);
	bSizer->ShowItems(true);
	this->Layout();
}

void GLFrame::OnScroll(wxScrollEvent& event) {
	m_canvas->OnScroll(m_scrollBar->GetThumbPosition());
}

void GLFrame::OnMousewheel(wxMouseEvent& event) {
	wxKeyboardState keyboard;
	int newScrollPos = event.GetWheelRotation() / MOUSEWHEELMAG;

	if (event.m_controlDown && event.m_altDown)
		m_canvas->recon->appendMinLight(newScrollPos);
	else if (event.m_controlDown) {
		m_canvas->recon->appendZoom(newScrollPos);
		m_canvas->recon->appendOffsets((event.GetX() - GetSize().x / 2) / SCROLLFACTOR * newScrollPos, (event.GetY() - GetSize().y / 2) / SCROLLFACTOR * newScrollPos);
	}
	else if (event.m_altDown)
		m_canvas->recon->appendMaxLight(newScrollPos);
	else {
		if (m_canvas->recon->getDataDisplay() == reconstruction) {
			m_canvas->recon->stepDistance(newScrollPos);
			m_canvas->recon->singleFrame();
		}
		else {
			newScrollPos += m_scrollBar->GetThumbPosition();
			if (newScrollPos < 0) newScrollPos = 0;
			if (newScrollPos > m_scrollBar->GetRange() - 1) newScrollPos = m_scrollBar->GetRange() - 1;
			m_scrollBar->SetThumbPosition(newScrollPos);
			m_canvas->OnScroll(newScrollPos);
		}
	}
	m_canvas->paint();
}

wxBEGIN_EVENT_TABLE(GLWindow, wxWindow)
EVT_MOUSEWHEEL(GLWindow::OnMousewheel)
EVT_CLOSE(GLWindow::onClose)
wxEND_EVENT_TABLE()

GLWindow::GLWindow(wxWindow *parent, bool vertical, struct SystemControl * Sys, wxString gainFile, wxString filename,
	const wxPoint& pos, const wxSize& size, long style)
	: wxDialog(parent, wxID_ANY, wxT("Select autofocus area with ctrl+mouse drag. Hit space once selected."), pos, size, style), m_canvas(NULL) {
	//Set up sizer to make the canvas take up the entire panel (wxWidgets handles garbage collection)
	wxBoxSizer* bSizer;
	bSizer = new wxBoxSizer(wxVERTICAL);

	//initialize the canvas to this object
	m_canvas = new CudaGLInCanvas(this, vertical, Sys, gainFile, filename, wxID_ANY, NULL, GetClientSize());
	bSizer->Add(m_canvas, 1, wxEXPAND | wxALL, 5);

	this->SetSizer(bSizer);
	this->Layout();
	bSizer->Fit(this);

	Show(true);
	Raise();
}

GLWindow::~GLWindow() {
	delete m_canvas;
}

void GLWindow::OnMousewheel(wxMouseEvent& event) {
	wxKeyboardState keyboard;
	int newScrollPos = event.GetWheelRotation() / MOUSEWHEELMAG;

	if (event.m_controlDown && event.m_altDown)
		m_canvas->recon->appendMinLight(newScrollPos);
	else if (event.m_controlDown) {
		m_canvas->recon->appendZoom(newScrollPos);
		m_canvas->recon->appendOffsets((event.GetX() - GetSize().x / 2) / SCROLLFACTOR * newScrollPos, (event.GetY() - GetSize().y / 2) / SCROLLFACTOR * newScrollPos);
	}
	else if (event.m_altDown)
		m_canvas->recon->appendMaxLight(newScrollPos);

	m_canvas->paint();
}

void GLWindow::onClose(wxCloseEvent& event) {
	Destroy();
}

wxBEGIN_EVENT_TABLE(CudaGLCanvas, wxGLCanvas)
EVT_PAINT(CudaGLCanvas::OnPaint)
EVT_CHAR(CudaGLCanvas::OnChar)
EVT_MOUSE_EVENTS(CudaGLCanvas::OnMouseEvent)
wxEND_EVENT_TABLE()

CudaGLCanvas::CudaGLCanvas(wxWindow *parent, wxStatusBar* status, struct SystemControl * Sys,
	wxWindowID id, int* gl_attrib, wxSize size)
	: wxGLCanvas(parent, id, gl_attrib, wxDefaultPosition, size, wxFULL_REPAINT_ON_RESIZE), m_status(status){
	// Explicitly create a new rendering context instance for this canvas.
	m_glRC = new wxGLContext(this);

	SetCurrent(*m_glRC);

	recon = new TomoRecon(GetSize().x, GetSize().y, Sys);
	recon->init();
}

CudaGLCanvas::~CudaGLCanvas(){
	delete recon;
	delete m_glRC;
}

void CudaGLCanvas::OnScroll(int index) {
	recon->setActiveProjection(index);
	recon->singleFrame();
	paint();
}

void CudaGLCanvas::OnPaint(wxPaintEvent& WXUNUSED(event)){
	// OnPaint handlers must always create a wxPaintDC.
	wxPaintDC(this);

	paint();
}

void CudaGLCanvas::paint(bool disChanged, wxTextCtrl* dis, wxSlider* zoom, wxStaticText* zLbl,
	wxSlider* window, wxStaticText* wLbl, wxSlider* level, wxStaticText* lLbl) {
	if (dis != NULL) distanceControl = dis;
	if (zoom != NULL) zoomSlider = zoom;
	if (zLbl != NULL) zoomLabel = zLbl;
	if (window != NULL) windowSlider = window;
	if (wLbl != NULL) windowLabel = wLbl;
	if (level != NULL) levelSlider = level;
	if (lLbl != NULL) levelLabel = lLbl;
	SetCurrent(*m_glRC);//tells opengl which buffers to use, mutliple windows fail without this

	if (!disChanged) {
		unsigned int window, level;
		recon->getLight(&level, &window);
		if(distanceControl) distanceControl->SetValue(wxString::Format(wxT("%.2f"), recon->getDistance()));
		if (zoomSlider) zoomSlider->SetValue(recon->getZoom());
		if (windowSlider) windowSlider->SetValue(window / WINLVLFACTOR);
		if (levelSlider) levelSlider->SetValue(level / WINLVLFACTOR);
		if (zoomLabel) zoomLabel->SetLabelText(wxString::Format(wxT("%.2f"), pow(ZOOMFACTOR, recon->getZoom())));
		if (windowLabel) windowLabel->SetLabelText(wxString::Format(wxT("%d"), window));
		if (levelLabel) levelLabel->SetLabelText(wxString::Format(wxT("%d"), level));
	}

	if (m_status) {
		int xOff, yOff;
		recon->getOffsets(&xOff, &yOff);
		m_status->SetStatusText(wxString::Format(wxT("X offset: %d px."), xOff), xOffset);
		m_status->SetStatusText(wxString::Format(wxT("Y offset: %d px."), yOff), yOffset);
	}

	recon->draw(GetSize().x, GetSize().y);

	SwapBuffers();
}

void CudaGLCanvas::OnMouseEvent(wxMouseEvent& event) {
	static int last_x, last_y, last_x_off, last_y_off;
	int this_x = event.GetX();
	int this_y = event.GetY();

	// Allow default processing to happen, or else the canvas cannot gain focus
	event.Skip();

	if (event.LeftDown()) {
		recon->getOffsets(&last_x_off, &last_y_off);
		last_x = this_x;
		last_y = this_y;
		
		if (event.m_controlDown)
			recon->setSelBoxStart(this_x, this_y);
		else recon->resetSelBox();
	}

	if (event.LeftIsDown())	{
		if(event.Dragging()){
			if (event.m_controlDown)
				recon->setSelBoxEnd(this_x, this_y);
			else 
				recon->setOffsets(last_x_off - (this_x - last_x), last_y_off - (this_y - last_y));
		}
	}

	paint();
}

void CudaGLCanvas::OnChar(wxKeyEvent& event){
	//Switch the derivative display
#ifdef USEITERATIVE
	static int errorIndex = 0;
	static int reconIndex = 0;
	if (event.GetKeyCode() == 32) {
		switch (recon->getDataDisplay()) {
		case error:
			recon->setDataDisplay(iterRecon);
			errorIndex = recon->getActiveProjection();
			recon->setActiveProjection(reconIndex);
			recon->setDisplay(no_der);
			recon->setLogView(false);
			((GLFrame*)GetParent())->showScrollBar(RECONSLICES, reconIndex);
			break;
		case iterRecon:
			recon->iterStep();
			recon->setDataDisplay(error);
			reconIndex = recon->getActiveProjection();
			recon->setActiveProjection(errorIndex);
			recon->setDisplay(no_der);
			//((GLFrame*)GetParent())->showScrollBar(RECONSLICES, reconIndex);
			((GLFrame*)GetParent())->showScrollBar(NUMVIEWS, errorIndex);

			/*recon->setDataDisplay(reconstruction);
			recon->setDisplay(mag_enhance);
			((GLFrame*)GetParent())->hideScrollBar();*/
			break;
		default:
			/*recon->setDataDisplay(error);
			recon->setDisplay(no_der);
			recon->setShowNegative(true);
			((GLFrame*)GetParent())->showScrollBar(NUMVIEWS, 0);
			recon->initIterative();*/
			//recon->setDataDisplay(iterRecon);
			((DTRMainWindow*)(GetParent()->GetParent()->GetParent()))->setDataDisplay((GLFrame*)GetParent(), iterRecon);
			//((GLFrame*)GetParent())->showScrollBar(RECONSLICES, 0);
			recon->initIterative();
			bool oldLog = recon->getLogView();
			recon->setLogView(false);
			for (int i = 0; i < 100; i++) {
				recon->iterStep();
				recon->singleFrame();
				recon->resetLight();
				paint();
			}
			recon->finalizeIter();
			recon->setLogView(oldLog);
			break;
		}

		recon->singleFrame();
		recon->resetLight();

		paint();
	}
#endif
#ifdef ENABLEZDER
	if (event.GetKeyCode() == 32) {
		switch (recon->getDisplay()) {
		case no_der:
			recon->setDisplay(mag_der);
			recon->setShowNegative(true);
			break;
		case mag_der:
			//recon->setDataDisplay(projections);
			recon->setShowNegative(true);
			recon->setDisplay(z_der_mag);
			break;
		case z_der_mag:
			recon->setShowNegative(true);
			recon->setDisplay(mag_der);
			break;
		/*case x_enhance:
			recon->setDisplay(mag_enhance);
			break;
		case mag_enhance:
			recon->setDisplay(y_enhance);
			break;
		case y_enhance:
			recon->setDisplay(both_enhance);
			break;
		case both_enhance:
			recon->setDisplay(no_der);
			//recon->setDisplay(der_x);
			break;
		case der_x:
			recon->setDisplay(der_y);
			break;
		case der_y:
			recon->setDisplay(slice_diff);
			break;
		case slice_diff:
			recon->setDisplay(square_mag);
			break;
		case square_mag:
			recon->setDisplay(der2_x);
			break;
		case der2_x:
			recon->setDisplay(der2_y);
			break;
		case der2_y:
			recon->setDisplay(der3_x);
			break;
		case der3_x:
			recon->setDisplay(der3_y);
			break;
		case der3_y:
			recon->setDisplay(no_der);
			break;*/
		default:
			recon->setDisplay(mag_der);
			break;
		}

		recon->singleFrame();
		recon->resetLight();

		paint();
	}
#endif //ENABLEZDER
}

wxBEGIN_EVENT_TABLE(CudaGLInCanvas, wxGLCanvas)
EVT_PAINT(CudaGLInCanvas::OnPaint)
EVT_MOUSE_EVENTS(CudaGLInCanvas::OnMouseEvent)
EVT_CHAR(CudaGLInCanvas::OnChar)
wxEND_EVENT_TABLE()

CudaGLInCanvas::CudaGLInCanvas(wxWindow *parent, bool vertical, struct SystemControl * Sys, wxString gainFile, wxString filename,
	wxWindowID id, int* gl_attrib, wxSize size)
	: wxGLCanvas(parent, id, gl_attrib, wxDefaultPosition, size, wxFULL_REPAINT_ON_RESIZE) {
	// Explicitly create a new rendering context instance for this canvas.
	m_glRC = new wxGLContext(this);

	SetCurrent(*m_glRC);

	recon = new TomoRecon(GetSize().x, GetSize().y, Sys);
	recon->init();
	recon->setInputVeritcal(vertical);
}

CudaGLInCanvas::~CudaGLInCanvas() {
	delete recon;
	delete m_glRC;
}

void CudaGLInCanvas::OnPaint(wxPaintEvent& WXUNUSED(event)) {
	// OnPaint handlers must always create a wxPaintDC.
	wxPaintDC(this);

	paint();
}

void CudaGLInCanvas::paint() {
	SetCurrent(*m_glRC);//tells opengl which buffers to use, mutliple windows fail without this
	recon->draw(GetSize().x, GetSize().y);
	SwapBuffers();
}

void CudaGLInCanvas::OnMouseEvent(wxMouseEvent& event) {
	static int last_x, last_y, last_x_off, last_y_off;
	int this_x = event.GetX();
	int this_y = event.GetY();

	// Allow default processing to happen, or else the canvas cannot gain focus
	event.Skip();

	if (event.LeftDown()) {
		if (event.m_controlDown) {
			switch (state) {
			case box1:
			case box2:
				recon->setSelBoxStart(this_x, this_y);
				break;
			case lower:
				recon->setLowerTick(this_x, this_y);
				break;
			case upper:
				recon->setUpperTick(this_x, this_y);
				break;
			}
		}
		last_x = this_x;
		last_y = this_y;
		recon->getOffsets(&last_x_off, &last_y_off);
	}

	if (event.LeftIsDown()) {
		if (event.Dragging()) {
			if (event.m_controlDown) {
				switch (state) {
				case box1:
				case box2:
					recon->setSelBoxEnd(this_x, this_y);
					break;
				case lower:
					recon->setLowerTick(this_x, this_y);
					break;
				case upper:
					recon->setUpperTick(this_x, this_y);
					break;
				}
			}
			else {
				recon->setOffsets(last_x_off - (this_x - last_x), last_y_off - (this_y - last_y));
			}
		}
	}

	paint();
}

void CudaGLInCanvas::OnChar(wxKeyEvent& event) {
	//pressing space advances state to next input
	if (event.GetKeyCode() == 32) {//32=space, enter is a system dialog reserved key
		switch (state) {
		case box1:
			if (recon->selBoxReady()) {
				state = box2;
				((GLWindow*)GetParent())->SetTitle(wxT("Select area of interest in the phantom with ctrl + mouse drag.Hit space once selected."));

				//transfer box data to temporary storage
				recon->getSelBoxRaw(&x1, &x2, &y1, &y2);
				recon->resetSelBox();
			}
		case box2:
			if (recon->selBoxReady()) {
				state = lower;
				((GLWindow*)GetParent())->SetTitle(wxT("Choose the lower bound on line pairs with ctrl+click. Hit space when done."));
			}
			break;
		case lower:
			if (recon->lowerTickReady()) {
				state = upper;
				((GLWindow*)GetParent())->SetTitle(wxT("Choose the upper bound on line pairs with ctrl+click. Hit space when done."));
			}
			break;
		case upper:
			if (recon->upperTickReady()) {
				//Close up and save (handled in parents)
				((GLWindow*)GetParent())->EndModal(wxID_OK);
			}
			break;
		}
	}

	paint();
}