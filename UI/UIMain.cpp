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

#ifdef PROFILER
	frame->Show(true);
	static unsigned s_pageAdded = 0;
	frame->m_auinotebook6->AddPage(frame->CreateNewPage(),
		wxString::Format
		(
			wxT("%u"),
			++s_pageAdded
		),
		true);
	wxCommandEvent test;
	frame->onContRun(test);
	exit(0);
#else
	frame->Show(true);
#endif

	return true;
}

// ----------------------------------------------------------------------------
// main frame
// ----------------------------------------------------------------------------

DTRMainWindow::DTRMainWindow(wxWindow* parent) : mainWindow(parent){
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
	darkFilepath = pConfig->Read(wxT("/darkFilepath"), wxT(""));
}

// event handlers
void DTRMainWindow::onNew(wxCommandEvent& WXUNUSED(event)) {
	static int s_pageAdded = 1;
	(*m_textCtrl8) << "Opening new tab titled: \"" << (int)s_pageAdded << "\"\n";

	//Step 1: Get and example file for get the path
#ifdef PROFILER
	wxString filename = wxT("C:\\Users\\jdean\\Desktop\\Patient18\\AcquiredImage1_0.raw");
#else
	wxFileDialog openFileDialog(this, _("Select one raw image file"), "", "",
		"Raw File (*.raw)|*.raw", wxFD_OPEN | wxFD_FILE_MUST_EXIST);

	if (openFileDialog.ShowModal() == wxID_CANCEL)
		return;

	wxString filename(openFileDialog.GetPath());
#endif

	m_auinotebook6->AddPage(CreateNewPage(filename), wxString::Format(wxT("%u"), s_pageAdded++), true);
	onContinuous();
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

	Sys->Geo.ZPitch = pConfig->ReadDouble(wxT("/sliceThickness"), 0.5f);
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
	struct SystemControl * Sys = new SystemControl;
	genSys(Sys);
	wxStreamToTextRedirector redirect(m_textCtrl8);
	return new GLFrame(m_auinotebook6, m_statusBar1, Sys, gainFilepath, darkFilepath, filename);
}

void DTRMainWindow::onOpen(wxCommandEvent& WXUNUSED(event)) {
	if (m_auinotebook6->GetCurrentPage() == m_panel10) {
		(*m_textCtrl8) << "Currently in console, cannot run. Open a new dataset with \"new\" (ctrl + n).\n";
		return;
	}

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	if (!recon->continuousMode) {
		(*m_textCtrl8) << "Open currently only works with reconstructions created with new, then run in continuous mode.\n";
		return;
	}

	wxFileDialog openFileDialog(this, _("Select one raw image file"), "", "",
		"Raw File (*.raw)|*.raw", wxFD_OPEN | wxFD_FILE_MUST_EXIST);

	if (openFileDialog.ShowModal() == wxID_CANCEL)
		return;

	wxString filename(openFileDialog.GetPath());
	
	wxStreamToTextRedirector redirect(m_textCtrl8);
	recon->TomoLoad(filename.mb_str());
	recon->currentDisplay = recon_images;
	currentFrame->m_canvas->paint();
}

void DTRMainWindow::onQuit(wxCommandEvent& WXUNUSED(event)){
	// true is to force the frame to close
	Close();
}

void DTRMainWindow::onProjectionView(wxCommandEvent& WXUNUSED(event)) {
	wxMessageBox(wxT("TODO"),
		wxT("TODO"),
		wxICON_INFORMATION | wxOK);
}

void DTRMainWindow::onReconstructionView(wxCommandEvent& WXUNUSED(event)) {
	wxMessageBox(wxT("TODO"),
		wxT("TODO"),
		wxICON_INFORMATION | wxOK);
}

void DTRMainWindow::onLogView(wxCommandEvent& WXUNUSED(event)) {
	if (m_auinotebook6->GetCurrentPage() == m_panel10) {
		(*m_textCtrl8) << "Currently in console, cannot run. Open a new dataset with \"new\" (ctrl + n).\n";
		return;
	}

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	recon->setLogView(!recon->getLogView());
	recon->resetLight();
	currentFrame->m_canvas->paint();
}

void DTRMainWindow::onResetFocus(wxCommandEvent& WXUNUSED(event)) {
	if (m_auinotebook6->GetCurrentPage() == m_panel10) {
		(*m_textCtrl8) << "Currently in console, cannot run. Open a new dataset with \"new\" (ctrl + n).\n";
		return;
	}

	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;

	recon->resetFocus();
}

void DTRMainWindow::onContinuous() {
	GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
	TomoRecon* recon = currentFrame->m_canvas->recon;
	int statusWidths[] = { -4, -1, -1, -1, -1 };

	wxStreamToTextRedirector redirect(m_textCtrl8);
	m_statusBar1->SetFieldsCount(5, statusWidths);
	recon->continuousMode = true;

	recon->singleFrame();
	recon->currentDisplay = recon_images;
	currentFrame->m_scrollBar->SetThumbPosition(0);
	currentFrame->m_canvas->OnScroll(0);
	currentFrame->m_scrollBar->Show(false);

	recon->resetFocus();

	currentFrame->m_canvas->paint();
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

void DTRMainWindow::onDarkSelect(wxCommandEvent& WXUNUSED(event)) {
	//Open files with raw extensions
	char temp[MAX_PATH];
	strncpy(temp, (const char*)darkFilepath.mb_str(), MAX_PATH - 1);
	OPENFILENAME ofn;
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = GetHWND();
	ofn.lpstrFilter = (LPCWSTR)"Raw files\0*.raw\0All Files\0*.*\0";
	ofn.lpstrFile = (LPWSTR)temp;
	ofn.nMaxFile = MAX_PATH;
	ofn.lpstrTitle = (LPCWSTR)"Select a dark file";
	ofn.Flags = OFN_DONTADDTORECENT | OFN_FILEMUSTEXIST;

	GetOpenFileNameA((LPOPENFILENAMEA)&ofn);

	darkFilepath = wxString::FromUTF8(temp);

	//Save filepath for next session
	wxConfigBase::Get()->Write(wxT("/darkFilePath"), darkFilepath);
}

void DTRMainWindow::onResList(wxCommandEvent& event) {
	if (resDialog == NULL) {
		resDialog = new DTRResDialog(this);
		resDialog->Show(true);
	}
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
	for (int i = 0; i < pConfig->Read(wxT("/resPhanItems"), 0l); i++)
	//if(pConfig->Read(wxT("/resPhanItems"), 0l) > 0)
	{
		//int i = 0;
		wxFileName filename = pConfig->Read(wxString::Format(wxT("/resPhanFile%d"), i));
		if (i == 0) {
			m_auinotebook6->AddPage(CreateNewPage(filename.GetFullPath()), wxString::Format(wxT("Geo Test %u"), i), true);
			onContinuous();
		}
		GLFrame* currentFrame = (GLFrame*)m_auinotebook6->GetCurrentPage();
		TomoRecon* recon = currentFrame->m_canvas->recon;

		if (i != 0) recon->TomoLoad(filename.GetFullPath().mb_str());

		if (data.empty()) recon->initTolerances(data, 1, offsets);
		
		recon->baseX = pConfig->Read(wxString::Format(wxT("/resPhanBoxLxF%d"), i), 0l);
		recon->baseY = pConfig->Read(wxString::Format(wxT("/resPhanBoxLyF%d"), i), 0l);
		recon->currX = pConfig->Read(wxString::Format(wxT("/resPhanBoxUxF%d"), i), 0l);
		recon->currY = pConfig->Read(wxString::Format(wxT("/resPhanBoxUyF%d"), i), 0l);
		recon->lowX = pConfig->Read(wxString::Format(wxT("/resPhanLowx%d"), i), 0l);
		recon->lowY = pConfig->Read(wxString::Format(wxT("/resPhanLowy%d"), i), 0l);
		recon->upX = pConfig->Read(wxString::Format(wxT("/resPhanUpx%d"), i), 0l);
		recon->upY = pConfig->Read(wxString::Format(wxT("/resPhanUpy%d"), i), 0l);
		recon->vertical = pConfig->Read(wxString::Format(wxT("/resPhanVert%d"), i), 0l) == 1;

		recon->setReconBox(0);
		recon->autoFocus(true);
		currentFrame->m_canvas->paint();
		recon->setReconBox(0);
		while (recon->autoFocus(false) == Tomo_OK) {
			recon->setReconBox(0);
			currentFrame->m_canvas->paint();
		}
		/*recon->autoGeo(true);
		currentFrame->m_canvas->paint();
		recon->setReconBox(0);
		while (recon->autoGeo(false) == Tomo_OK) {
			recon->setReconBox(0);
			currentFrame->m_canvas->paint();
		}*/

		//switch from autofocus box to area of interest
		recon->baseX = pConfig->Read(wxString::Format(wxT("/resPhanBoxLx%d"), i), 0l);
		recon->baseY = pConfig->Read(wxString::Format(wxT("/resPhanBoxLy%d"), i), 0l);
		recon->currX = pConfig->Read(wxString::Format(wxT("/resPhanBoxUx%d"), i), 0l);
		recon->currY = pConfig->Read(wxString::Format(wxT("/resPhanBoxUy%d"), i), 0l);
		recon->setReconBox(0);

		recon->singleFrame();
		currentFrame->m_canvas->paint();

		int output = 0;
		std::ofstream FILE;
		m_statusBar1->SetStatusText(filename.GetFullPath());
		FILE.open(wxString::Format(wxT("%s\\testResults.txt"), filename.GetPath()).mb_str());
		recon->testTolerances(data, true);
		while (recon->testTolerances(data, false) == Tomo_OK) {
			currentFrame->m_canvas->paint();
			FILE << data[output].name << ", " << data[output].numViewsChanged << ", " << data[output].viewsChanged << ", " 
				<< data[output].offset << ", " << data[output].thisDir << ", " << data[output].phantomData << "\n";
			output++;
		}
		FILE.close();
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

void DTRMainWindow::onPageChange(wxCommandEvent& WXUNUSED(event)) {
	wxString toolTip = m_auinotebook6->GetPageToolTip(m_auinotebook6->GetPageIndex(m_auinotebook6->GetCurrentPage()));
	(*m_textCtrl8) << toolTip;
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

	cuda(DeviceReset());//only reset here where we know all windows are finished
}

// ----------------------------------------------------------------------------
// Config frame handling
// ----------------------------------------------------------------------------

DTRConfigDialog::DTRConfigDialog(wxWindow* parent) : configDialog(parent){
	//load all values from previously saved settings
	wxConfigBase *pConfig = wxConfigBase::Get();

	orientation->SetSelection(pConfig->ReadLong(wxT("/orientation"), 0l) == 0l ? 0 : 1);
	rotationEnabled->SetSelection(pConfig->ReadLong(wxT("/rotationEnabled"), 0l) == 0l ? 0 : 1);
	sliceThickness->SetValue(wxString::Format(wxT("%.1f"), pConfig->ReadDouble(wxT("/sliceThickness"), 0.5f)));
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
	orientation->SetSelection(cJSON_GetObjectItem(root, "orientation")->type == cJSON_False ? 0 : 1);
	rotationEnabled->SetSelection(cJSON_GetObjectItem(root, "rotationEnabled")->type == cJSON_False ? 0 : 1);
	sliceThickness->SetValue(wxString::Format(wxT("%.1f"), cJSON_GetObjectItem(root, "sliceThickness")->valuedouble));
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
	sliceThickness->SetValue(wxString::Format(wxT("%.1f"), atof(data_in)));
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
	rotationEnabled->SetSelection(atoi(data_in) == 0 ? 0 : 1);
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
	orientation->SetSelection(atoi(data_in) == 0 ? 0 : 1);
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
	if (orientation->GetSelection() == 0) cJSON_AddFalseToObject(root, "orientation");
	else cJSON_AddTrueToObject(root, "orientation");
	if (rotationEnabled->GetSelection() == 0) cJSON_AddFalseToObject(root, "rotationEnabled");
	else cJSON_AddTrueToObject(root, "rotationEnabled");

	sliceThickness->GetLineText(0).ToDouble(&parsedDouble);
	cJSON_AddNumberToObject(root, "sliceThickness", parsedDouble);
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

	pConfig->Write(wxT("/orientation"), orientation->GetSelection() == 0 ? 0l : 1l);
	pConfig->Write(wxT("/rotationEnabled"), rotationEnabled->GetSelection() == 0 ? 0l : 1l);

	if (!sliceThickness->GetLineText(0).ToDouble(&parsedDouble)) {
		wxMessageBox(wxT("Invalid input in text box: \"Thickness of reconstruciton slice\"."),
			wxT("Invlaid input"),
			wxICON_STOP | wxOK);
		return Tomo_input_err;
	}
	else pConfig->Write(wxT("/sliceThickness"), parsedDouble);

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
	wxFileDialog openFileDialog(this, _("Open raw image file"), "", "",
			"Raw File (*.raw)|*.raw", wxFD_OPEN | wxFD_FILE_MUST_EXIST);

	if (openFileDialog.ShowModal() == wxID_CANCEL)
		return;

	int vertical = wxMessageBox(wxT("Is the selected input vertical? (no=horizontal)"), 
		wxT("Input orientation"),
		wxICON_INFORMATION | wxYES | wxNO);

	struct SystemControl * Sys = new SystemControl;
	((DTRMainWindow*)GetParent())->genSys(Sys);
	frame = new GLWindow(this, vertical == wxYES, Sys, ((DTRMainWindow*)GetParent())->gainFilepath, ((DTRMainWindow*)GetParent())->darkFilepath, openFileDialog.GetPath());
	int res = frame->ShowModal();

	if (res == wxID_OK) {
		//User successfully completed the dialog interaction
		wxString file = openFileDialog.GetPath();
		int index = m_listCtrl->FindItem(-1, file);
		if (index == wxNOT_FOUND)
			index = m_listCtrl->InsertItem(0, file);

		float scale = frame->m_canvas->recon->scale;
		float xOff = frame->m_canvas->recon->xOff;
		float yOff = frame->m_canvas->recon->yOff;
		float innerOffx = (frame->m_canvas->recon->width - Sys->Proj.Nx / scale) / 2;
		float innerOffy = (frame->m_canvas->recon->height - Sys->Proj.Ny / scale) / 2;
		m_listCtrl->SetItem(index, 1, wxString::Format(wxT("%d"), (int)((frame->m_canvas->recon->baseX - innerOffx) * scale + xOff)));
		m_listCtrl->SetItem(index, 2, wxString::Format(wxT("%d"), (int)((frame->m_canvas->recon->baseY - innerOffy) * scale + yOff)));
		m_listCtrl->SetItem(index, 3, wxString::Format(wxT("%d"), (int)((frame->m_canvas->recon->currX - innerOffx) * scale + xOff)));
		m_listCtrl->SetItem(index, 4, wxString::Format(wxT("%d"), (int)((frame->m_canvas->recon->currY - innerOffy) * scale + yOff)));
		m_listCtrl->SetItem(index, 5, wxString::Format(wxT("%d"), (int)((frame->m_canvas->recon->lowX - innerOffx) * scale + xOff)));
		m_listCtrl->SetItem(index, 6, wxString::Format(wxT("%d"), (int)((frame->m_canvas->recon->lowY - innerOffy) * scale + yOff)));
		m_listCtrl->SetItem(index, 7, wxString::Format(wxT("%d"), (int)((frame->m_canvas->recon->upX - innerOffx) * scale + xOff)));
		m_listCtrl->SetItem(index, 8, wxString::Format(wxT("%d"), (int)((frame->m_canvas->recon->upY - innerOffy) * scale + yOff)));
		m_listCtrl->SetItem(index, 9, vertical == wxYES ? wxT("Yes") : wxT("No"));
		m_listCtrl->SetItem(index, 10, wxString::Format(wxT("%d"), (int)((frame->m_canvas->recon->constants.baseXr - innerOffx) * scale + xOff)));
		m_listCtrl->SetItem(index, 11, wxString::Format(wxT("%d"), (int)((frame->m_canvas->recon->constants.baseYr - innerOffy) * scale + yOff)));
		m_listCtrl->SetItem(index, 12, wxString::Format(wxT("%d"), (int)((frame->m_canvas->recon->constants.currXr - innerOffx) * scale + xOff)));
		m_listCtrl->SetItem(index, 13, wxString::Format(wxT("%d"), (int)((frame->m_canvas->recon->constants.currYr - innerOffy) * scale + yOff)));
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

	((DTRMainWindow*)GetParent())->resDialog = NULL;
	Close(true);
}

void DTRResDialog::onCancel(wxCommandEvent& event) {
	((DTRMainWindow*)GetParent())->resDialog = NULL;
	Close(true);
}

DTRResDialog::~DTRResDialog() {

}

//---------------------------------------------------------------------------
// GLFrame
//---------------------------------------------------------------------------

wxBEGIN_EVENT_TABLE(GLFrame, wxPanel)
EVT_SCROLL(GLFrame::OnScroll)
EVT_MOUSEWHEEL(GLFrame::OnMousewheel)
wxEND_EVENT_TABLE()

GLFrame::GLFrame(wxAuiNotebook *frame, wxStatusBar* status, struct SystemControl * Sys, wxString gainFile, wxString darkFile, wxString filename,
	const wxPoint& pos, const wxSize& size, long style)
	: wxPanel(frame, wxID_ANY, pos, size), m_canvas(NULL), m_status(status){
	//Set up sizer to make the canvas take up the entire panel (wxWidgets handles garbage collection)
	wxBoxSizer* bSizer;
	bSizer = new wxBoxSizer(wxVERTICAL);

	//initialize the canvas to this object
	m_canvas = new CudaGLCanvas(this, status, Sys, gainFile, darkFile, filename, wxID_ANY, NULL, GetClientSize());
	bSizer->Add(m_canvas, 1, wxEXPAND | wxALL, 5);

	m_scrollBar = new wxScrollBar(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxSB_HORIZONTAL);
	m_scrollBar->SetScrollbar(0, 1, 7, 1);
	bSizer->Add(m_scrollBar, 0, wxALL | wxEXPAND, 5);

	this->SetSizer(bSizer);
	this->Layout();
	bSizer->Fit(this);

	// Show the frame
	Show(true);
	Raise();//grab attention when the frame has finished rendering
}

GLFrame::~GLFrame(){
	delete m_canvas;
}

void GLFrame::OnScroll(wxScrollEvent& event) {
	m_canvas->OnScroll(m_scrollBar->GetThumbPosition());
}

void GLFrame::OnMousewheel(wxMouseEvent& event) {
	wxKeyboardState keyboard;
	//GetKeyboardState()
	int newScrollPos = event.GetWheelRotation() / 120;
	if (event.m_controlDown && event.m_altDown) {
		m_canvas->recon->lightOff += newScrollPos;
	}
	else if (event.m_controlDown) {
		m_canvas->recon->zoom += newScrollPos;
		if (m_canvas->recon->zoom < 0) m_canvas->recon->zoom = 0;
		m_canvas->recon->xOff += (event.GetX() - GetSize().x / 2)*m_canvas->recon->scale / 10 * newScrollPos;//- GetScreenPosition().x
		m_canvas->recon->yOff += (event.GetY() - GetSize().y / 2)*m_canvas->recon->scale / 10 * newScrollPos;// - GetScreenPosition().y
	}
	else if (event.m_altDown) {
		m_canvas->recon->light += newScrollPos;
	}
	else {
		if (m_canvas->recon->continuousMode) {
			m_canvas->recon->distance += newScrollPos*m_canvas->recon->Sys->Geo.ZPitch;
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

//---------------------------------------------------------------------------
// GLWindow
//---------------------------------------------------------------------------

wxBEGIN_EVENT_TABLE(GLWindow, wxWindow)
EVT_MOUSEWHEEL(GLWindow::OnMousewheel)
EVT_CLOSE(GLWindow::onClose)
wxEND_EVENT_TABLE()

GLWindow::GLWindow(wxWindow *parent, bool vertical, struct SystemControl * Sys, wxString gainFile, wxString darkFile, wxString filename,
	const wxPoint& pos, const wxSize& size, long style)
	: wxDialog(parent, wxID_ANY, wxT("Select autofocus area with ctrl+mouse drag. Hit space once selected."), pos, size, style), m_canvas(NULL) {
	//Set up sizer to make the canvas take up the entire panel (wxWidgets handles garbage collection)
	wxBoxSizer* bSizer;
	bSizer = new wxBoxSizer(wxVERTICAL);

	//initialize the canvas to this object
	m_canvas = new CudaGLInCanvas(this, vertical, Sys, gainFile, darkFile, filename, wxID_ANY, NULL, GetClientSize());
	bSizer->Add(m_canvas, 1, wxEXPAND | wxALL, 5);

	this->SetSizer(bSizer);
	this->Layout();
	bSizer->Fit(this);
}

GLWindow::~GLWindow() {
	delete m_canvas;
}

void GLWindow::OnMousewheel(wxMouseEvent& event) {
	wxKeyboardState keyboard;
	//GetKeyboardState()
	int newScrollPos = event.GetWheelRotation() / 120;
	if (event.m_controlDown && event.m_altDown) {
		m_canvas->recon->lightOff += newScrollPos;
		m_canvas->paint();
	}
	else if (event.m_controlDown) {
		m_canvas->recon->zoom += newScrollPos;
		if (m_canvas->recon->zoom < 0) m_canvas->recon->zoom = 0;
		m_canvas->paint();
		m_canvas->recon->xOff += (event.GetX() - GetSize().x / 2)*m_canvas->recon->scale / 10 * newScrollPos;//- GetScreenPosition().x
		m_canvas->recon->yOff += (event.GetY() - GetSize().y / 2)*m_canvas->recon->scale / 10 * newScrollPos;// - GetScreenPosition().y
	}
	else if (event.m_altDown) {
		m_canvas->recon->light += newScrollPos;
		//if (m_canvas->recon->light < 0) m_canvas->recon->light = 0;
		m_canvas->paint();
	}
}

void GLWindow::onClose(wxCloseEvent& event) {
	Destroy();
}

//---------------------------------------------------------------------------
// CudaGLCanvas
//---------------------------------------------------------------------------

wxBEGIN_EVENT_TABLE(CudaGLCanvas, wxGLCanvas)
EVT_PAINT(CudaGLCanvas::OnPaint)
EVT_CHAR(CudaGLCanvas::OnChar)
EVT_MOUSE_EVENTS(CudaGLCanvas::OnMouseEvent)
wxEND_EVENT_TABLE()

CudaGLCanvas::CudaGLCanvas(wxWindow *parent, wxStatusBar* status, struct SystemControl * Sys, wxString gainFile, wxString darkFile, wxString filename, 
	wxWindowID id, int* gl_attrib, wxSize size)
	: wxGLCanvas(parent, id, gl_attrib, wxDefaultPosition, size, wxFULL_REPAINT_ON_RESIZE), m_status(status){
	// Explicitly create a new rendering context instance for this canvas.
	m_glRC = new wxGLContext(this);

	SetCurrent(*m_glRC);

	recon = new TomoRecon(GetSize().x, GetSize().y, Sys);
	recon->init((const char*)gainFile.mb_str(), (const char*)darkFile.mb_str(), (const char*)filename.mb_str());

	recon->sliceIndex = 0;//initialization in recon.h doesn't work for some reason
	recon->zoom = 0;
}

CudaGLCanvas::~CudaGLCanvas(){
	delete recon;
	delete m_glRC;
}

void CudaGLCanvas::OnScroll(int index) {
	imageIndex = index;
	paint();
}

void CudaGLCanvas::OnEvent(wxCommandEvent& WXUNUSED(event)) {
	if (recon->initialized) {
		paint();
	}
}

void CudaGLCanvas::OnPaint(wxPaintEvent& WXUNUSED(event)){
	// This is a dummy, to avoid an endless succession of paint messages.
	// OnPaint handlers must always create a wxPaintDC.
	wxPaintDC(this);

	if (recon->initialized) {
		paint();
	}
}

void CudaGLCanvas::paint() {
	SetCurrent(*m_glRC);//tells opengl which buffers to use, mutliple windows fail without this
	int width = GetSize().x;
	int height = GetSize().y;
	recon->display(width, height);
	recon->map();

	if (recon->continuousMode) {
		m_status->SetStatusText(wxString::Format(wxT("Zoom: %.2fx"), pow(ZOOMFACTOR, recon->zoom)), scaleNum);
		m_status->SetStatusText(wxString::Format(wxT("X offset: %d px."), recon->xOff), xOffset);
		m_status->SetStatusText(wxString::Format(wxT("Y offset: %d px."), recon->yOff), yOffset);
		m_status->SetStatusText(wxString::Format(wxT("Detector distance: %.2f mm."), recon->distance), zPosition);
	}

	recon->test(imageIndex);

	recon->unmap();

	recon->blit();
	recon->swap();

	SwapBuffers();
}

void CudaGLCanvas::OnMouseEvent(wxMouseEvent& event) {
	static float last_x, last_y, last_x_off, last_y_off;
	float this_x = event.GetX();
	float this_y = event.GetY();

	// Allow default processing to happen, or else the canvas cannot gain focus
	// (for key events).
	event.Skip();

	if (event.LeftDown()) {
		last_x = this_x;
		last_y = this_y;
		
		last_x_off = recon->xOff;
		last_y_off = recon->yOff;
		if (event.m_controlDown) {
			recon->constants.baseXr = recon->D2I(this_x, true);
			recon->constants.baseYr = recon->D2I(this_y, false);
			recon->constants.currXr = -1;
			recon->constants.currYr = -1;
		}
	}

	if (event.LeftIsDown())	{
		if(event.Dragging()){
			if (!event.m_controlDown) {
				recon->xOff = last_x_off - (this_x - last_x)*recon->scale;
				recon->yOff = last_y_off - (this_y - last_y)*recon->scale;
			}
			else {
				recon->constants.currXr = recon->D2I(this_x, true);
				recon->constants.currYr = recon->D2I(this_y, false);
			}
		}
	}

	if (event.LeftUp()) {
		if (recon->constants.baseXr >= 0 && recon->constants.currXr >= 0) {
			//if they're greater than 0, the box was clicked and dragged successfully
			recon->autoFocus(true);
			paint();
			while (recon->autoFocus(false) == Tomo_OK) paint();
			if (event.m_altDown) {
				//dependent on a focused image
				recon->autoGeo(true);
				paint();
				while (recon->autoGeo(false) == Tomo_OK) paint();
			}
			
			//cleanup
			recon->constants.baseXr = -1;
			recon->constants.currXr = -1;
			recon->lowXr = -1;
			recon->upXr = -1;
			recon->singleFrame();
		}
	}

	paint();
}

void CudaGLCanvas::OnChar(wxKeyEvent& event){
	//Switch the derivative display
	if (event.GetKeyCode() == 32) {
		switch (recon->derDisplay) {
		case no_der:
			recon->derDisplay = der_x;
			break;
		case der_x:
			recon->derDisplay = der_y;
			break;
		case der_y:
			recon->derDisplay = slice_diff;
			break;
		case slice_diff:
			recon->derDisplay = square_mag;
			break;
		case square_mag:
			recon->derDisplay = der2_x;
			break;
		case der2_x:
			recon->derDisplay = der2_y;
			break;
		case der2_y:
			recon->derDisplay = der3_x;
			break;
		case der3_x:
			recon->derDisplay = der3_y;
			break;
		case der3_y:
			recon->derDisplay = no_der;
			break;
		}

		recon->constants.baseXr = 3 * recon->Sys->Recon.Nx / 4;
		recon->constants.baseYr = 3 * recon->Sys->Recon.Ny / 4;
		recon->constants.currXr = recon->Sys->Recon.Nx / 4;
		recon->constants.currYr = recon->Sys->Recon.Ny / 4;

		recon->singleFrame();
		recon->autoLight();

		recon->constants.baseXr = -1;
		recon->constants.baseYr = -1;
		recon->constants.currXr = -1;
		recon->constants.currYr = -1;

		paint();
	}
}

//---------------------------------------------------------------------------
// CudaGLInCanvas
//---------------------------------------------------------------------------

wxBEGIN_EVENT_TABLE(CudaGLInCanvas, wxGLCanvas)
EVT_PAINT(CudaGLInCanvas::OnPaint)
EVT_MOUSE_EVENTS(CudaGLInCanvas::OnMouseEvent)
EVT_CHAR(CudaGLInCanvas::OnChar)
wxEND_EVENT_TABLE()

CudaGLInCanvas::CudaGLInCanvas(wxWindow *parent, bool vertical, struct SystemControl * Sys, wxString gainFile, wxString darkFile, wxString filename,
	wxWindowID id, int* gl_attrib, wxSize size)
	: wxGLCanvas(parent, id, gl_attrib, wxDefaultPosition, size, wxFULL_REPAINT_ON_RESIZE) {
	// Explicitly create a new rendering context instance for this canvas.
	m_glRC = new wxGLContext(this);

	SetCurrent(*m_glRC);

	recon = new TomoRecon(GetSize().x, GetSize().y, Sys);
	recon->init((const char*)gainFile.mb_str(), (const char*)darkFile.mb_str(), (const char*)filename.mb_str());
	recon->vertical = vertical;
}

CudaGLInCanvas::~CudaGLInCanvas() {
	delete recon;
	delete m_glRC;
}

void CudaGLInCanvas::OnEvent(wxCommandEvent& WXUNUSED(event)) {
	if (recon->initialized) {
		paint();
	}
}

void CudaGLInCanvas::OnPaint(wxPaintEvent& WXUNUSED(event)) {
	// This is a dummy, to avoid an endless succession of paint messages.
	// OnPaint handlers must always create a wxPaintDC.
	wxPaintDC(this);

	if (recon->initialized) {
		paint();
	}
}

void CudaGLInCanvas::paint() {
	SetCurrent(*m_glRC);//tells opengl which buffers to use, mutliple windows fail without this
	int width = GetSize().x;
	int height = GetSize().y;
	recon->display(width, height);
	recon->map();

	recon->test(imageIndex);

	recon->unmap();

	recon->blit();
	recon->swap();

	SwapBuffers();
}

void CudaGLInCanvas::OnMouseEvent(wxMouseEvent& event) {
	static float last_x, last_y, last_x_off, last_y_off;
	float this_x = event.GetX();
	float this_y = event.GetY();

	// Allow default processing to happen, or else the canvas cannot gain focus
	// (for key events).
	event.Skip();

	if (event.LeftDown()) {
		if (event.m_controlDown) {
			switch (state) {
			case box1:
			case box2:
				recon->baseX = this_x;
				recon->baseY = this_y;
				break;
			case lower:
				recon->lowX = this_x;
				recon->lowY = this_y;
				break;
			case upper:
				recon->upX = this_x;
				recon->upY = this_y;
				break;
			}
			paint();
		}
		last_x = this_x;
		last_y = this_y;
		last_x_off = recon->xOff;
		last_y_off = recon->yOff;
	}

	if (event.LeftIsDown()) {
		if (event.Dragging()) {
			if (event.m_controlDown) {
				switch (state) {
				case box1:
				case box2:
					recon->currX = this_x;
					recon->currY = this_y;
					break;
				case lower:
					recon->lowX = this_x;
					recon->lowY = this_y;
					break;
				case upper:
					recon->upX = this_x;
					recon->upY = this_y;
					break;
				}
			}
			else {
				recon->xOff = last_x_off - (this_x - last_x)*recon->scale;
				recon->yOff = last_y_off - (this_y - last_y)*recon->scale;
			}
			paint();
		}
	}
}

void CudaGLInCanvas::OnChar(wxKeyEvent& event) {
	//pressing space advances state to next input
	if (event.GetKeyCode() == 32) {//32=space, enter is a system dialog reserved key
		switch (state) {
		case box1:
			if (recon->baseX >= 0 && recon->currX >= 0) {
				state = box2;
				((GLWindow*)GetParent())->SetTitle(wxT("Select area of interest in the phantom with ctrl + mouse drag.Hit space once selected."));

				//transfer box data to temporary storage internal to recon
				recon->constants.baseXr = recon->baseX;
				recon->constants.baseYr = recon->baseY;
				recon->constants.currXr = recon->currX;
				recon->constants.currYr = recon->currY;
				recon->baseX = -1;
				recon->baseY = -1;
				recon->currX = -1;
				recon->currY = -1;
			}
		case box2:
			if (recon->baseX >= 0 && recon->currX >= 0) {
				state = lower;
				((GLWindow*)GetParent())->SetTitle(wxT("Choose the lower bound on line pairs with ctrl+click. Hit space when done."));
			}
			break;
		case lower:
			if (recon->lowX >= 0) {
				state = upper;
				((GLWindow*)GetParent())->SetTitle(wxT("Choose the upper bound on line pairs with ctrl+click. Hit space when done."));
			}
			break;
		case upper:
			if (recon->upX >= 0) {
				//Close up and save (handled in parents)
				((GLWindow*)GetParent())->EndModal(wxID_OK);
			}
			break;
		}
	}
}