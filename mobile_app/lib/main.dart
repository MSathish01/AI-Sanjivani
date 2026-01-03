import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'screens/home_screen.dart';
import 'screens/assessment_screen.dart';
import 'screens/results_screen.dart';
import 'screens/history_screen.dart';
import 'providers/health_provider.dart';
import 'providers/language_provider.dart';
import 'services/database_service.dart';
import 'utils/app_localizations.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Initialize database
  await DatabaseService.instance.database;
  
  runApp(const AISanjivaniApp());
}

class AISanjivaniApp extends StatelessWidget {
  const AISanjivaniApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => LanguageProvider()),
        ChangeNotifierProvider(create: (_) => HealthProvider()),
      ],
      child: Consumer<LanguageProvider>(
        builder: (context, languageProvider, child) {
          return MaterialApp(
            title: 'AI-Sanjivani',
            debugShowCheckedModeBanner: false,
            
            // Localization
            locale: languageProvider.currentLocale,
            localizationsDelegates: const [
              AppLocalizations.delegate,
              GlobalMaterialLocalizations.delegate,
              GlobalWidgetsLocalizations.delegate,
              GlobalCupertinoLocalizations.delegate,
            ],
            supportedLocales: const [
              Locale('en', 'US'), // English
              Locale('hi', 'IN'), // Hindi
              Locale('mr', 'IN'), // Marathi
              Locale('ta', 'IN'), // Tamil
            ],
            
            // Theme optimized for rural users
            theme: ThemeData(
              primarySwatch: Colors.green,
              primaryColor: const Color(0xFF2E7D32), // Medical green
              scaffoldBackgroundColor: const Color(0xFFF8F9FA),
              
              // Large, accessible fonts
              textTheme: const TextTheme(
                headlineLarge: TextStyle(
                  fontSize: 32,
                  fontWeight: FontWeight.bold,
                  color: Color(0xFF1B5E20),
                ),
                headlineMedium: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.w600,
                  color: Color(0xFF2E7D32),
                ),
                bodyLarge: TextStyle(
                  fontSize: 18,
                  color: Color(0xFF424242),
                ),
                bodyMedium: TextStyle(
                  fontSize: 16,
                  color: Color(0xFF616161),
                ),
              ),
              
              // High contrast buttons for visibility
              elevatedButtonTheme: ElevatedButtonThemeData(
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF2E7D32),
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(
                    horizontal: 32,
                    vertical: 16,
                  ),
                  textStyle: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w600,
                  ),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
              ),
              
              // Card theme for assessment cards
              cardTheme: CardTheme(
                elevation: 4,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16),
                ),
                margin: const EdgeInsets.symmetric(
                  horizontal: 16,
                  vertical: 8,
                ),
              ),
              
              // App bar theme
              appBarTheme: const AppBarTheme(
                backgroundColor: Color(0xFF2E7D32),
                foregroundColor: Colors.white,
                elevation: 0,
                centerTitle: true,
                titleTextStyle: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            
            // Routes
            initialRoute: '/',
            routes: {
              '/': (context) => const HomeScreen(),
              '/assessment': (context) => const AssessmentScreen(),
              '/results': (context) => const ResultsScreen(),
              '/history': (context) => const HistoryScreen(),
            },
          );
        },
      ),
    );
  }
}

/// Language Provider for multilingual support
class LanguageProvider extends ChangeNotifier {
  Locale _currentLocale = const Locale('en', 'US');
  
  Locale get currentLocale => _currentLocale;
  
  final List<Map<String, dynamic>> _supportedLanguages = [
    {
      'code': 'en',
      'country': 'US',
      'name': 'English',
      'nativeName': 'English',
    },
    {
      'code': 'hi',
      'country': 'IN',
      'name': 'Hindi',
      'nativeName': 'हिंदी',
    },
    {
      'code': 'mr',
      'country': 'IN',
      'name': 'Marathi',
      'nativeName': 'मराठी',
    },
    {
      'code': 'ta',
      'country': 'IN',
      'name': 'Tamil',
      'nativeName': 'தமிழ்',
    },
  ];
  
  List<Map<String, dynamic>> get supportedLanguages => _supportedLanguages;
  
  LanguageProvider() {
    _loadSavedLanguage();
  }
  
  void _loadSavedLanguage() async {
    final prefs = await SharedPreferences.getInstance();
    final languageCode = prefs.getString('language_code') ?? 'en';
    final countryCode = prefs.getString('country_code') ?? 'US';
    
    _currentLocale = Locale(languageCode, countryCode);
    notifyListeners();
  }
  
  void changeLanguage(String languageCode, String countryCode) async {
    _currentLocale = Locale(languageCode, countryCode);
    
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('language_code', languageCode);
    await prefs.setString('country_code', countryCode);
    
    notifyListeners();
  }
  
  String getCurrentLanguageName() {
    final current = _supportedLanguages.firstWhere(
      (lang) => lang['code'] == _currentLocale.languageCode,
      orElse: () => _supportedLanguages[0],
    );
    return current['nativeName'];
  }
}