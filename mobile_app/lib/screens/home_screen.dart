import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:connectivity_plus/connectivity_plus.dart';

import '../providers/language_provider.dart';
import '../providers/health_provider.dart';
import '../utils/app_localizations.dart';
import '../widgets/language_selector.dart';
import '../widgets/offline_indicator.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with TickerProviderStateMixin {
  late AnimationController _animationController;
  late Animation<double> _fadeAnimation;
  bool _isOffline = false;

  @override
  void initState() {
    super.initState();
    
    // Initialize animations
    _animationController = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    );
    
    _fadeAnimation = Tween<double>(
      begin: 0.0,
      end: 1.0,
    ).animate(CurvedAnimation(
      parent: _animationController,
      curve: Curves.easeInOut,
    ));
    
    _animationController.forward();
    
    // Check connectivity
    _checkConnectivity();
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  void _checkConnectivity() async {
    final connectivityResult = await Connectivity().checkConnectivity();
    setState(() {
      _isOffline = connectivityResult == ConnectivityResult.none;
    });
  }

  @override
  Widget build(BuildContext context) {
    final localizations = AppLocalizations.of(context);
    final languageProvider = Provider.of<LanguageProvider>(context);
    
    return Scaffold(
      appBar: AppBar(
        title: Text(localizations.appTitle),
        actions: [
          IconButton(
            icon: const Icon(Icons.language),
            onPressed: () => _showLanguageSelector(context),
          ),
          IconButton(
            icon: const Icon(Icons.history),
            onPressed: () => Navigator.pushNamed(context, '/history'),
          ),
        ],
      ),
      body: FadeTransition(
        opacity: _fadeAnimation,
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Offline indicator
              if (_isOffline) const OfflineIndicator(),
              
              const SizedBox(height: 20),
              
              // Welcome section
              _buildWelcomeSection(localizations),
              
              const SizedBox(height: 32),
              
              // Main action buttons
              _buildActionButtons(context, localizations),
              
              const SizedBox(height: 32),
              
              // Information cards
              _buildInfoCards(localizations),
              
              const SizedBox(height: 32),
              
              // Emergency contact
              _buildEmergencyContact(localizations),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildWelcomeSection(AppLocalizations localizations) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          children: [
            // AI-Sanjivani logo/icon
            Container(
              width: 80,
              height: 80,
              decoration: BoxDecoration(
                color: Theme.of(context).primaryColor,
                shape: BoxShape.circle,
              ),
              child: const Icon(
                Icons.health_and_safety,
                size: 40,
                color: Colors.white,
              ),
            ),
            
            const SizedBox(height: 16),
            
            Text(
              localizations.welcomeTitle,
              style: Theme.of(context).textTheme.headlineMedium,
              textAlign: TextAlign.center,
            ),
            
            const SizedBox(height: 8),
            
            Text(
              localizations.welcomeSubtitle,
              style: Theme.of(context).textTheme.bodyLarge,
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildActionButtons(BuildContext context, AppLocalizations localizations) {
    return Column(
      children: [
        // Start Health Assessment
        SizedBox(
          width: double.infinity,
          height: 60,
          child: ElevatedButton.icon(
            onPressed: () => Navigator.pushNamed(context, '/assessment'),
            icon: const Icon(Icons.medical_services, size: 24),
            label: Text(localizations.startAssessment),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF2E7D32),
            ),
          ),
        ),
        
        const SizedBox(height: 16),
        
        // View History
        SizedBox(
          width: double.infinity,
          height: 60,
          child: OutlinedButton.icon(
            onPressed: () => Navigator.pushNamed(context, '/history'),
            icon: const Icon(Icons.history, size: 24),
            label: Text(localizations.viewHistory),
            style: OutlinedButton.styleFrom(
              side: BorderSide(color: Theme.of(context).primaryColor, width: 2),
              foregroundColor: Theme.of(context).primaryColor,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildInfoCards(AppLocalizations localizations) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          localizations.howItWorks,
          style: Theme.of(context).textTheme.headlineMedium,
        ),
        
        const SizedBox(height: 16),
        
        // Step cards
        _buildStepCard(
          1,
          localizations.step1Title,
          localizations.step1Description,
          Icons.record_voice_over,
          Colors.blue,
        ),
        
        _buildStepCard(
          2,
          localizations.step2Title,
          localizations.step2Description,
          Icons.psychology,
          Colors.orange,
        ),
        
        _buildStepCard(
          3,
          localizations.step3Title,
          localizations.step3Description,
          Icons.health_and_safety,
          Colors.green,
        ),
      ],
    );
  }

  Widget _buildStepCard(
    int stepNumber,
    String title,
    String description,
    IconData icon,
    Color color,
  ) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Row(
          children: [
            // Step number circle
            Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                color: color,
                shape: BoxShape.circle,
              ),
              child: Center(
                child: Text(
                  stepNumber.toString(),
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                    fontSize: 18,
                  ),
                ),
              ),
            ),
            
            const SizedBox(width: 16),
            
            // Content
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    description,
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                ],
              ),
            ),
            
            // Icon
            Icon(
              icon,
              color: color,
              size: 32,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildEmergencyContact(AppLocalizations localizations) {
    return Card(
      color: Colors.red.shade50,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Row(
              children: [
                Icon(
                  Icons.emergency,
                  color: Colors.red.shade700,
                  size: 24,
                ),
                const SizedBox(width: 8),
                Text(
                  localizations.emergencyContact,
                  style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                    fontWeight: FontWeight.bold,
                    color: Colors.red.shade700,
                  ),
                ),
              ],
            ),
            
            const SizedBox(height: 12),
            
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildEmergencyButton('108', localizations.emergency),
                _buildEmergencyButton('104', localizations.healthHelpline),
                _buildEmergencyButton('102', localizations.ambulance),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildEmergencyButton(String number, String label) {
    return Column(
      children: [
        ElevatedButton(
          onPressed: () {
            // In a real app, this would make a phone call
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(content: Text('Calling $number...')),
            );
          },
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.red.shade700,
            shape: const CircleBorder(),
            padding: const EdgeInsets.all(16),
          ),
          child: Text(
            number,
            style: const TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
              color: Colors.white,
            ),
          ),
        ),
        const SizedBox(height: 4),
        Text(
          label,
          style: Theme.of(context).textTheme.bodySmall,
          textAlign: TextAlign.center,
        ),
      ],
    );
  }

  void _showLanguageSelector(BuildContext context) {
    showModalBottomSheet(
      context: context,
      builder: (context) => const LanguageSelector(),
    );
  }
}