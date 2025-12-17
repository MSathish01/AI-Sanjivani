import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/health_assessment_provider.dart';
import '../providers/speech_provider.dart';
import '../widgets/symptom_selector.dart';
import '../widgets/voice_input_button.dart';
import '../widgets/demographic_form.dart';
import '../widgets/progress_indicator.dart';
import '../utils/app_localizations.dart';
import '../utils/colors.dart';

class AssessmentScreen extends StatefulWidget {
  const AssessmentScreen({super.key});

  @override
  State<AssessmentScreen> createState() => _AssessmentScreenState();
}

class _AssessmentScreenState extends State<AssessmentScreen> {
  final PageController _pageController = PageController();
  int _currentStep = 0;
  final int _totalSteps = 3;

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final localizations = AppLocalizations.of(context);
    
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: AppColors.textPrimary),
          onPressed: () => Navigator.of(context).pop(),
        ),
        title: Text(
          localizations.translate('health_assessment'),
          style: const TextStyle(
            color: AppColors.textPrimary,
            fontWeight: FontWeight.w600,
          ),
        ),
        centerTitle: true,
      ),
      body: Column(
        children: [
          // Progress indicator
          Padding(
            padding: const EdgeInsets.all(20.0),
            child: CustomProgressIndicator(
              currentStep: _currentStep,
              totalSteps: _totalSteps,
            ),
          ),
          
          // Page content
          Expanded(
            child: PageView(
              controller: _pageController,
              onPageChanged: (index) {
                setState(() {
                  _currentStep = index;
                });
              },
              children: [
                _buildDemographicStep(localizations),
                _buildSymptomStep(localizations),
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

  Widget _buildDemographicStep(AppLocalizations localizations) {
    return Padding(
      padding: const EdgeInsets.all(20.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildStepHeader(
            localizations.translate('basic_information'),
            localizations.translate('basic_info_subtitle'),
            Icons.person,
          ),
          
          const SizedBox(height: 30),
          
          Expanded(
            child: Consumer<HealthAssessmentProvider>(
              builder: (context, provider, child) {
                return DemographicForm(
                  onDataChanged: (data) {
                    provider.updateDemographicData(data);
                  },
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSymptomStep(AppLocalizations localizations) {
    return Padding(
      padding: const EdgeInsets.all(20.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildStepHeader(
            localizations.translate('symptoms'),
            localizations.translate('symptoms_subtitle'),
            Icons.medical_services,
          ),
          
          const SizedBox(height: 20),
          
          // Voice input section
          Consumer<SpeechProvider>(
            builder: (context, speechProvider, child) {
              return VoiceInputButton(
                isListening: speechProvider.isListening,
                onPressed: () => _handleVoiceInput(speechProvider),
                recognizedText: speechProvider.recognizedText,
              );
            },
          ),
          
          const SizedBox(height: 30),
          
          // Symptom selector
          Expanded(
            child: Consumer<HealthAssessmentProvider>(
              builder: (context, provider, child) {
                return SymptomSelector(
                  selectedSymptoms: provider.selectedSymptoms,
                  onSymptomsChanged: (symptoms) {
                    provider.updateSymptoms(symptoms);
                  },
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildConfirmationStep(AppLocalizations localizations) {
    return Padding(
      padding: const EdgeInsets.all(20.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildStepHeader(
            localizations.translate('review_information'),
            localizations.translate('review_subtitle'),
            Icons.check_circle,
          ),
          
          const SizedBox(height: 30),
          
          Expanded(
            child: Consumer<HealthAssessmentProvider>(
              builder: (context, provider, child) {
                return _buildReviewContent(localizations, provider);
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStepHeader(String title, String subtitle, IconData icon) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: AppColors.primary.withOpacity(0.1),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: AppColors.primary,
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(
              icon,
              color: Colors.white,
              size: 24,
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: AppColors.textPrimary,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  subtitle,
                  style: const TextStyle(
                    fontSize: 14,
                    color: AppColors.textSecondary,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildReviewContent(AppLocalizations localizations, HealthAssessmentProvider provider) {
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Demographic information
          _buildReviewSection(
            localizations.translate('basic_information'),
            [
              '${localizations.translate('age')}: ${provider.demographicData['age'] ?? 'Not specified'}',
              '${localizations.translate('gender')}: ${provider.demographicData['gender'] ?? 'Not specified'}',
              if (provider.demographicData['isPregnant'] == true)
                localizations.translate('pregnant'),
              if (provider.demographicData['hasDiabetes'] == true)
                localizations.translate('has_diabetes'),
              if (provider.demographicData['hasHypertension'] == true)
                localizations.translate('has_hypertension'),
            ],
            Icons.person,
          ),
          
          const SizedBox(height: 24),
          
          // Symptoms
          _buildReviewSection(
            localizations.translate('symptoms'),
            provider.selectedSymptoms.isNotEmpty
                ? provider.selectedSymptoms
                : [localizations.translate('no_symptoms_selected')],
            Icons.medical_services,
          ),
          
          const SizedBox(height: 24),
          
          // Disclaimer
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: AppColors.warning.withOpacity(0.1),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: AppColors.warning.withOpacity(0.3)),
            ),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(
                  Icons.info,
                  color: AppColors.warning,
                  size: 20,
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    localizations.translate('assessment_disclaimer'),
                    style: const TextStyle(
                      fontSize: 12,
                      color: AppColors.textSecondary,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildReviewSection(String title, List<String> items, IconData icon) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: AppColors.primary, size: 20),
              const SizedBox(width: 8),
              Text(
                title,
                style: const TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                  color: AppColors.textPrimary,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          ...items.map((item) => Padding(
            padding: const EdgeInsets.only(bottom: 4),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('• ', style: TextStyle(color: AppColors.textSecondary)),
                Expanded(
                  child: Text(
                    item,
                    style: const TextStyle(
                      fontSize: 14,
                      color: AppColors.textSecondary,
                    ),
                  ),
                ),
              ],
            ),
          )).toList(),
        ],
      ),
    );
  }

  Widget _buildNavigationButtons(AppLocalizations localizations) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: const BoxDecoration(
        color: AppColors.surface,
        border: Border(
          top: BorderSide(color: AppColors.border, width: 1),
        ),
      ),
      child: Row(
        children: [
          // Back button
          if (_currentStep > 0)
            Expanded(
              child: OutlinedButton(
                onPressed: _goToPreviousStep,
                style: OutlinedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  side: const BorderSide(color: AppColors.primary),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: Text(
                  localizations.translate('back'),
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                    color: AppColors.primary,
                  ),
                ),
              ),
            ),
          
          if (_currentStep > 0) const SizedBox(width: 16),
          
          // Next/Submit button
          Expanded(
            flex: _currentStep == 0 ? 1 : 1,
            child: Consumer<HealthAssessmentProvider>(
              builder: (context, provider, child) {
                final isLastStep = _currentStep == _totalSteps - 1;
                final canProceed = _canProceedToNextStep(provider);
                
                return ElevatedButton(
                  onPressed: canProceed
                      ? (isLastStep ? _submitAssessment : _goToNextStep)
                      : null,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.primary,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    elevation: 0,
                  ),
                  child: provider.isLoading
                      ? const SizedBox(
                          height: 20,
                          width: 20,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                          ),
                        )
                      : Text(
                          isLastStep
                              ? localizations.translate('get_assessment')
                              : localizations.translate('next'),
                          style: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  bool _canProceedToNextStep(HealthAssessmentProvider provider) {
    switch (_currentStep) {
      case 0:
        return provider.demographicData['age'] != null &&
               provider.demographicData['gender'] != null;
      case 1:
        return provider.selectedSymptoms.isNotEmpty;
      case 2:
        return true;
      default:
        return false;
    }
  }

  void _goToNextStep() {
    if (_currentStep < _totalSteps - 1) {
      _pageController.nextPage(
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeInOut,
      );
    }
  }

  void _goToPreviousStep() {
    if (_currentStep > 0) {
      _pageController.previousPage(
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeInOut,
      );
    }
  }

  void _handleVoiceInput(SpeechProvider speechProvider) async {
    if (speechProvider.isListening) {
      await speechProvider.stopListening();
    } else {
      await speechProvider.startListening();
      
      // Process recognized symptoms
      if (speechProvider.recognizedText.isNotEmpty) {
        final healthProvider = Provider.of<HealthAssessmentProvider>(context, listen: false);
        final extractedSymptoms = await speechProvider.extractSymptomsFromText(
          speechProvider.recognizedText,
        );
        
        if (extractedSymptoms.isNotEmpty) {
          healthProvider.addSymptomsFromVoice(extractedSymptoms);
        }
      }
    }
  }

  void _submitAssessment() async {
    final provider = Provider.of<HealthAssessmentProvider>(context, listen: false);
    
    try {
      await provider.performAssessment();
      
      if (mounted) {
        Navigator.pushReplacementNamed(context, '/result');
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Assessment failed: ${e.toString()}'),
            backgroundColor: AppColors.error,
          ),
        );
      }
    }
  }
}