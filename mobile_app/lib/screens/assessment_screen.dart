import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:speech_to_text/speech_to_text.dart';
import 'package:permission_handler/permission_handler.dart';

import '../providers/health_provider.dart';
import '../utils/app_localizations.dart';
import '../widgets/symptom_selector.dart';
import '../widgets/voice_input_widget.dart';
import '../widgets/assessment_progress.dart';

class AssessmentScreen extends StatefulWidget {
  const AssessmentScreen({super.key});

  @override
  State<AssessmentScreen> createState() => _AssessmentScreenState();
}

class _AssessmentScreenState extends State<AssessmentScreen> {
  final PageController _pageController = PageController();
  int _currentStep = 0;
  final int _totalSteps = 4;

  // Form data
  int? _age;
  String? _gender;
  List<String> _selectedSymptoms = [];
  bool _isVoiceMode = false;

  // Voice recognition
  final SpeechToText _speechToText = SpeechToText();
  bool _speechEnabled = false;
  bool _speechListening = false;

  @override
  void initState() {
    super.initState();
    _initSpeech();
  }

  void _initSpeech() async {
    final status = await Permission.microphone.request();
    if (status == PermissionStatus.granted) {
      _speechEnabled = await _speechToText.initialize();
      setState(() {});
    }
  }

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final localizations = AppLocalizations.of(context);
    
    return Scaffold(
      appBar: AppBar(
        title: Text(localizations.healthAssessment),
        elevation: 0,
      ),
      body: Column(
        children: [
          // Progress indicator
          AssessmentProgress(
            currentStep: _currentStep,
            totalSteps: _totalSteps,
          ),
          
          // Main content
          Expanded(
            child: PageView(
              controller: _pageController,
              onPageChanged: (index) {
                setState(() {
                  _currentStep = index;
                });
              },
              children: [
                _buildWelcomeStep(localizations),
                _buildBasicInfoStep(localizations),
                _buildSymptomsStep(localizations),
                _buildConfirmationStep(localizations),
              ],
            ),
          ),
          
          // Navigation buttons
          _buildNavigationButtons(localizations),
        ],
      ),
    );
  }

  Widget _buildWelcomeStep(AppLocalizations localizations) {
    return Padding(
      padding: const EdgeInsets.all(24.0),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.health_and_safety,
            size: 100,
            color: Theme.of(context).primaryColor,
          ),
          
          const SizedBox(height: 32),
          
          Text(
            localizations.assessmentWelcome,
            style: Theme.of(context).textTheme.headlineMedium,
            textAlign: TextAlign.center,
          ),
          
          const SizedBox(height: 16),
          
          Text(
            localizations.assessmentInstructions,
            style: Theme.of(context).textTheme.bodyLarge,
            textAlign: TextAlign.center,
          ),
          
          const SizedBox(height: 32),
          
          // Input mode selection
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                children: [
                  Text(
                    localizations.selectInputMode,
                    style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  
                  const SizedBox(height: 16),
                  
                  Row(
                    children: [
                      Expanded(
                        child: _buildModeButton(
                          localizations.textInput,
                          Icons.keyboard,
                          !_isVoiceMode,
                          () => setState(() => _isVoiceMode = false),
                        ),
                      ),
                      
                      const SizedBox(width: 16),
                      
                      Expanded(
                        child: _buildModeButton(
                          localizations.voiceInput,
                          Icons.mic,
                          _isVoiceMode,
                          _speechEnabled 
                            ? () => setState(() => _isVoiceMode = true)
                            : null,
                        ),
                      ),
                    ],
                  ),
                  
                  if (!_speechEnabled)
                    Padding(
                      padding: const EdgeInsets.only(top: 8.0),
                      child: Text(
                        localizations.voiceNotAvailable,
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: Colors.orange,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildModeButton(
    String label,
    IconData icon,
    bool isSelected,
    VoidCallback? onTap,
  ) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: isSelected 
            ? Theme.of(context).primaryColor.withOpacity(0.1)
            : Colors.grey.shade100,
          border: Border.all(
            color: isSelected 
              ? Theme.of(context).primaryColor
              : Colors.grey.shade300,
            width: 2,
          ),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          children: [
            Icon(
              icon,
              size: 32,
              color: isSelected 
                ? Theme.of(context).primaryColor
                : (onTap != null ? Colors.grey.shade600 : Colors.grey.shade400),
            ),
            const SizedBox(height: 8),
            Text(
              label,
              style: TextStyle(
                fontWeight: FontWeight.w600,
                color: isSelected 
                  ? Theme.of(context).primaryColor
                  : (onTap != null ? Colors.grey.shade700 : Colors.grey.shade400),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildBasicInfoStep(AppLocalizations localizations) {
    return Padding(
      padding: const EdgeInsets.all(24.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            localizations.basicInformation,
            style: Theme.of(context).textTheme.headlineMedium,
          ),
          
          const SizedBox(height: 32),
          
          // Age input
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    localizations.age,
                    style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  
                  const SizedBox(height: 12),
                  
                  TextFormField(
                    keyboardType: TextInputType.number,
                    decoration: InputDecoration(
                      hintText: localizations.enterAge,
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                      prefixIcon: const Icon(Icons.cake),
                    ),
                    onChanged: (value) {
                      _age = int.tryParse(value);
                    },
                  ),
                ],
              ),
            ),
          ),
          
          const SizedBox(height: 16),
          
          // Gender selection
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    localizations.gender,
                    style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  
                  const SizedBox(height: 12),
                  
                  Row(
                    children: [
                      Expanded(
                        child: _buildGenderButton(
                          localizations.male,
                          'M',
                          Icons.male,
                        ),
                      ),
                      
                      const SizedBox(width: 16),
                      
                      Expanded(
                        child: _buildGenderButton(
                          localizations.female,
                          'F',
                          Icons.female,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildGenderButton(String label, String value, IconData icon) {
    final isSelected = _gender == value;
    
    return GestureDetector(
      onTap: () => setState(() => _gender = value),
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: isSelected 
            ? Theme.of(context).primaryColor.withOpacity(0.1)
            : Colors.grey.shade100,
          border: Border.all(
            color: isSelected 
              ? Theme.of(context).primaryColor
              : Colors.grey.shade300,
            width: 2,
          ),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              icon,
              color: isSelected 
                ? Theme.of(context).primaryColor
                : Colors.grey.shade600,
            ),
            const SizedBox(width: 8),
            Text(
              label,
              style: TextStyle(
                fontWeight: FontWeight.w600,
                color: isSelected 
                  ? Theme.of(context).primaryColor
                  : Colors.grey.shade700,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSymptomsStep(AppLocalizations localizations) {
    return Padding(
      padding: const EdgeInsets.all(24.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            localizations.symptoms,
            style: Theme.of(context).textTheme.headlineMedium,
          ),
          
          const SizedBox(height: 16),
          
          Text(
            _isVoiceMode 
              ? localizations.voiceInstructions
              : localizations.symptomInstructions,
            style: Theme.of(context).textTheme.bodyMedium,
          ),
          
          const SizedBox(height: 24),
          
          Expanded(
            child: _isVoiceMode 
              ? VoiceInputWidget(
                  onSymptomsDetected: (symptoms) {
                    setState(() {
                      _selectedSymptoms = symptoms;
                    });
                  },
                )
              : SymptomSelector(
                  selectedSymptoms: _selectedSymptoms,
                  onSymptomsChanged: (symptoms) {
                    setState(() {
                      _selectedSymptoms = symptoms;
                    });
                  },
                ),
          ),
        ],
      ),
    );
  }

  Widget _buildConfirmationStep(AppLocalizations localizations) {
    return Padding(
      padding: const EdgeInsets.all(24.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            localizations.confirmDetails,
            style: Theme.of(context).textTheme.headlineMedium,
          ),
          
          const SizedBox(height: 24),
          
          // Summary cards
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildSummaryRow(
                    localizations.age,
                    _age?.toString() ?? localizations.notProvided,
                    Icons.cake,
                  ),
                  
                  const Divider(),
                  
                  _buildSummaryRow(
                    localizations.gender,
                    _gender == 'M' 
                      ? localizations.male 
                      : _gender == 'F' 
                        ? localizations.female 
                        : localizations.notProvided,
                    _gender == 'M' ? Icons.male : Icons.female,
                  ),
                  
                  const Divider(),
                  
                  _buildSummaryRow(
                    localizations.symptoms,
                    _selectedSymptoms.isNotEmpty 
                      ? _selectedSymptoms.join(', ')
                      : localizations.noSymptomsSelected,
                    Icons.medical_services,
                  ),
                ],
              ),
            ),
          ),
          
          const SizedBox(height: 24),
          
          // Assessment button
          SizedBox(
            width: double.infinity,
            height: 60,
            child: ElevatedButton.icon(
              onPressed: _canProceedToAssessment() ? _performAssessment : null,
              icon: const Icon(Icons.psychology),
              label: Text(localizations.analyzeSymptoms),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSummaryRow(String label, String value, IconData icon) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Row(
        children: [
          Icon(icon, color: Theme.of(context).primaryColor),
          const SizedBox(width: 12),
          Text(
            '$label: ',
            style: Theme.of(context).textTheme.bodyLarge?.copyWith(
              fontWeight: FontWeight.w600,
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: Theme.of(context).textTheme.bodyLarge,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildNavigationButtons(AppLocalizations localizations) {
    return Container(
      padding: const EdgeInsets.all(16.0),
      child: Row(
        children: [
          if (_currentStep > 0)
            Expanded(
              child: OutlinedButton(
                onPressed: _previousStep,
                child: Text(localizations.previous),
              ),
            ),
          
          if (_currentStep > 0) const SizedBox(width: 16),
          
          Expanded(
            child: ElevatedButton(
              onPressed: _canProceedToNext() ? _nextStep : null,
              child: Text(
                _currentStep == _totalSteps - 1 
                  ? localizations.analyze
                  : localizations.next,
              ),
            ),
          ),
        ],
      ),
    );
  }

  bool _canProceedToNext() {
    switch (_currentStep) {
      case 0:
        return true; // Welcome step
      case 1:
        return _age != null && _gender != null;
      case 2:
        return _selectedSymptoms.isNotEmpty;
      case 3:
        return _canProceedToAssessment();
      default:
        return false;
    }
  }

  bool _canProceedToAssessment() {
    return _age != null && 
           _gender != null && 
           _selectedSymptoms.isNotEmpty;
  }

  void _nextStep() {
    if (_currentStep < _totalSteps - 1) {
      _pageController.nextPage(
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeInOut,
      );
    } else {
      _performAssessment();
    }
  }

  void _previousStep() {
    if (_currentStep > 0) {
      _pageController.previousPage(
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeInOut,
      );
    }
  }

  void _performAssessment() async {
    final healthProvider = Provider.of<HealthProvider>(context, listen: false);
    
    // Show loading
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => const Center(
        child: CircularProgressIndicator(),
      ),
    );
    
    try {
      // Perform assessment
      await healthProvider.performAssessment(
        age: _age!,
        gender: _gender!,
        symptoms: _selectedSymptoms,
      );
      
      // Navigate to results
      if (mounted) {
        Navigator.of(context).pop(); // Close loading dialog
        Navigator.pushReplacementNamed(context, '/results');
      }
    } catch (e) {
      if (mounted) {
        Navigator.of(context).pop(); // Close loading dialog
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Assessment failed: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }
}