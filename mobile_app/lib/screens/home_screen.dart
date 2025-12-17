import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/app_state_provider.dart';
import '../widgets/language_selector.dart';
import '../widgets/large_action_button.dart';
import '../widgets/info_card.dart';
import '../utils/app_localizations.dart';
import '../utils/colors.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final localizations = AppLocalizations.of(context);
    final appState = Provider.of<AppStateProvider>(context);
    
    return Scaffold(
      backgroundColor: AppColors.background,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Header with logo and language selector
              _buildHeader(context, localizations, appState),
              
              const SizedBox(height: 30),
              
              // Welcome message
              _buildWelcomeMessage(localizations),
              
              const SizedBox(height: 40),
              
              // Main action buttons
              Expanded(
                child: _buildActionButtons(context, localizations),
              ),
              
              const SizedBox(height: 20),
              
              // Info cards
              _buildInfoCards(localizations),
              
              const SizedBox(height: 20),
              
              // Emergency contact
              _buildEmergencyContact(localizations),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeader(BuildContext context, AppLocalizations localizations, AppStateProvider appState) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        // Logo and app name
        Row(
          children: [
            Container(
              width: 50,
              height: 50,
              decoration: BoxDecoration(
                color: AppColors.primary,
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Icon(
                Icons.health_and_safety,
                color: Colors.white,
                size: 30,
              ),
            ),
            const SizedBox(width: 12),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'AI-Sanjivani',
                  style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                    fontWeight: FontWeight.bold,
                    color: AppColors.textPrimary,
                  ),
                ),
                Text(
                  localizations.translate('app_subtitle'),
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: AppColors.textSecondary,
                  ),
                ),
              ],
            ),
          ],
        ),
        
        // Language selector
        LanguageSelector(
          currentLanguage: appState.currentLanguage,
          onLanguageChanged: (language) {
            appState.setLanguage(language);
          },
        ),
      ],
    );
  }

  Widget _buildWelcomeMessage(AppLocalizations localizations) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [AppColors.primary.withOpacity(0.1), AppColors.secondary.withOpacity(0.1)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.primary.withOpacity(0.2)),
      ),
      child: Column(
        children: [
          Icon(
            Icons.waving_hand,
            size: 40,
            color: AppColors.primary,
          ),
          const SizedBox(height: 12),
          Text(
            localizations.translate('welcome_message'),
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w600,
              color: AppColors.textPrimary,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 8),
          Text(
            localizations.translate('welcome_subtitle'),
            style: const TextStyle(
              fontSize: 14,
              color: AppColors.textSecondary,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  Widget _buildActionButtons(BuildContext context, AppLocalizations localizations) {
    return Column(
      children: [
        // Start Health Check button
        LargeActionButton(
          icon: Icons.medical_services,
          title: localizations.translate('start_health_check'),
          subtitle: localizations.translate('check_symptoms'),
          color: AppColors.primary,
          onTap: () {
            Navigator.pushNamed(context, '/assessment');
          },
        ),
        
        const SizedBox(height: 20),
        
        // View History button
        LargeActionButton(
          icon: Icons.history,
          title: localizations.translate('view_history'),
          subtitle: localizations.translate('past_assessments'),
          color: AppColors.secondary,
          onTap: () {
            Navigator.pushNamed(context, '/history');
          },
        ),
        
        const SizedBox(height: 20),
        
        // Emergency button
        LargeActionButton(
          icon: Icons.emergency,
          title: localizations.translate('emergency'),
          subtitle: localizations.translate('call_108'),
          color: AppColors.error,
          onTap: () {
            _showEmergencyDialog(context, localizations);
          },
        ),
      ],
    );
  }

  Widget _buildInfoCards(AppLocalizations localizations) {
    return Row(
      children: [
        Expanded(
          child: InfoCard(
            icon: Icons.offline_bolt,
            title: localizations.translate('works_offline'),
            subtitle: localizations.translate('no_internet_needed'),
            color: AppColors.success,
          ),
        ),
        const SizedBox(width: 16),
        Expanded(
          child: InfoCard(
            icon: Icons.language,
            title: localizations.translate('multilingual'),
            subtitle: localizations.translate('hindi_marathi_support'),
            color: AppColors.info,
          ),
        ),
      ],
    );
  }

  Widget _buildEmergencyContact(AppLocalizations localizations) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.error.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.error.withOpacity(0.3)),
      ),
      child: Row(
        children: [
          Icon(
            Icons.phone_in_talk,
            color: AppColors.error,
            size: 24,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  localizations.translate('emergency_helpline'),
                  style: const TextStyle(
                    fontWeight: FontWeight.w600,
                    color: AppColors.textPrimary,
                  ),
                ),
                Text(
                  '108 - ${localizations.translate('ambulance_service')}',
                  style: const TextStyle(
                    color: AppColors.textSecondary,
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),
          Icon(
            Icons.arrow_forward_ios,
            color: AppColors.error,
            size: 16,
          ),
        ],
      ),
    );
  }

  void _showEmergencyDialog(BuildContext context, AppLocalizations localizations) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          title: Row(
            children: [
              Icon(Icons.emergency, color: AppColors.error),
              const SizedBox(width: 8),
              Text(localizations.translate('emergency')),
            ],
          ),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(localizations.translate('emergency_message')),
              const SizedBox(height: 16),
              _buildEmergencyContact(
                '108',
                localizations.translate('ambulance_service'),
                Icons.local_hospital,
              ),
              const SizedBox(height: 8),
              _buildEmergencyContact(
                '104',
                localizations.translate('health_helpline'),
                Icons.phone,
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: Text(localizations.translate('close')),
            ),
          ],
        );
      },
    );
  }

  Widget _buildEmergencyContact(String number, String service, IconData icon) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        children: [
          Icon(icon, color: AppColors.primary, size: 20),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  number,
                  style: const TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                  ),
                ),
                Text(
                  service,
                  style: const TextStyle(
                    fontSize: 12,
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
}