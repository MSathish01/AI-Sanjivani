import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'package:flutter_localizations/flutter_localizations.dart';

import 'providers/app_state_provider.dart';
import 'providers/health_assessment_provider.dart';
import 'providers/speech_provider.dart';
import 'screens/splash_screen.dart';
import 'screens/home_screen.dart';
import 'screens/assessment_screen.dart';
import 'screens/result_screen.dart';
import 'screens/history_screen.dart';
import 'utils/app_localizations.dart';
import 'utils/theme.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Set preferred orientations
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
  ]);
  
  // Set system UI overlay style
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.dark,
    ),
  );
  
  runApp(const AISanjivaniApp());
}

class AISanjivaniApp extends StatelessWidget {
  const AISanjivaniApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => AppStateProvider()),
        ChangeNotifierProvider(create: (_) => HealthAssessmentProvider()),
        ChangeNotifierProvider(create: (_) => SpeechProvider()),
      ],
      child: Consumer<AppStateProvider>(
        builder: (context, appState, child) {
          return MaterialApp(
            title: 'AI-Sanjivani',
            debugShowCheckedModeBanner: false,
            
            // Theme
            theme: AppTheme.lightTheme,
            darkTheme: AppTheme.darkTheme,
            themeMode: ThemeMode.light, // Always light for rural users
            
            // Localization
            locale: appState.currentLocale,
            supportedLocales: const [
              Locale('en', 'US'), // English
              Locale('hi', 'IN'), // Hindi
              Locale('mr', 'IN'), // Marathi
            ],
            localizationsDelegates: const [
              AppLocalizations.delegate,
              GlobalMaterialLocalizations.delegate,
              GlobalWidgetsLocalizations.delegate,
              GlobalCupertinoLocalizations.delegate,
            ],
            
            // Routes
            initialRoute: '/',
            routes: {
              '/': (context) => const SplashScreen(),
              '/home': (context) => const HomeScreen(),
              '/assessment': (context) => const AssessmentScreen(),
              '/result': (context) => const ResultScreen(),
              '/history': (context) => const HistoryScreen(),
            },
            
            // Accessibility
            builder: (context, child) {
              return MediaQuery(
                data: MediaQuery.of(context).copyWith(
                  textScaleFactor: 1.2, // Larger text for rural users
                ),
                child: child!,
              );
            },
          );
        },
      ),
    );
  }
}