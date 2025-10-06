#!/bin/bash
# Football AI System - Phase 4: React Frontend Setup
# Modern React 18 with TypeScript, Tailwind CSS, and Advanced Features

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Check if running as footballai user
check_user() {
    if [ "$USER" != "footballai" ]; then
        error "Bitte als footballai Benutzer ausführen (use: su - footballai)"
        exit 1
    fi
}

# Navigate to frontend directory
setup_environment() {
    log "🎨 Starte Frontend Setup..."
    cd ~/football-ai-system/frontend
}

# Initialize React app with TypeScript
init_react_app() {
    log "📦 Initialisiere React App mit TypeScript..."
    
    # Check if React app already exists
    if [ ! -f "package.json" ]; then
        # Create React app with TypeScript template
        npx create-react-app . --template typescript --use-npm
        log "✅ React App mit TypeScript erstellt"
    else
        log "✅ React App existiert bereits"
    fi
}

# Install core dependencies
install_core_dependencies() {
    log "🔧 Installiere Core Dependencies..."
    
    # Core libraries
    npm install @tanstack/react-query@^5.8.4 axios@^1.6.2
    
    # State management
    npm install zustand@^4.4.7 @reduxjs/toolkit@^1.9.7 react-redux@^8.1.3
    
    # Routing
    npm install react-router-dom@^6.18.0 @types/react-router-dom@^5.3.3
    
    # Forms and validation
    npm install react-hook-form@^7.48.2 @hookform/resolvers@^3.3.2 yup@^1.3.3
    
    # Real-time communication
    npm install socket.io-client@^4.7.4 @types/socket.io-client@^3.0.0
    
    # Date handling
    npm install date-fns@^2.30.0 @date-io/date-fns@^2.17.0
    
    # Utilities
    npm install lodash@^4.17.21 @types/lodash@^4.14.202 classnames@^2.3.2 react-toastify@^9.1.3
    
    # Animations
    npm install framer-motion@^10.16.5 react-spring@^9.7.3
    
    log "✅ Core Dependencies installiert"
}

# Install UI and styling dependencies
install_ui_dependencies() {
    log "🎨 Installiere UI und Styling Dependencies..."
    
    # Tailwind CSS
    npm install -D tailwindcss@^3.3.6 @tailwindcss/forms@^0.5.7 @tailwindcss/typography@^0.5.10 autoprefixer@^10.4.16 postcss@^8.4.32
    
    # Headless UI components
    npm install @headlessui/react@^1.7.17 @heroicons/react@^2.0.18
    
    # Styled components
    npm install styled-components@^6.1.1 @types/styled-components@^5.1.32
    
    # Icons
    npm install @fortawesome/fontawesome-free@^6.4.2 @fortawesome/react-fontawesome@^0.2.0 @fortawesome/fontawesome-svg-core@^6.4.2 @fortawesome/free-solid-svg-icons@^6.4.2 @fortawesome/free-brands-svg-icons@^6.4.2
    
    log "✅ UI Dependencies installiert"
}

# Install chart and visualization dependencies
install_chart_dependencies() {
    log "📊 Installiere Chart und Visualization Dependencies..."
    
    # Charts
    npm install recharts@^2.10.1 victory@^36.6.11 @types/d3@^7.4.3 d3@^7.8.5
    npm install react-chartjs-2@^5.2.0 chart.js@^4.4.0
    
    # Grid and table libraries
    npm install @mui/x-data-grid@^6.18.1
    
    log "✅ Chart Dependencies installiert"
}

# Install development dependencies
install_dev_dependencies() {
    log "🛠️ Installiere Development Dependencies..."
    
    # TypeScript and type definitions
    npm install -D @types/node@^16.18.68 @types/react@^18.2.42 @types/react-dom@^18.2.17
    
    # Testing
    npm install -D @types/jest@^27.5.2 @testing-library/react@^13.4.0 @testing-library/jest-dom@^5.16.5
    
    # Code quality
    npm install -D eslint@^8.54.0 @typescript-eslint/parser@^6.12.0 @typescript-eslint/eslint-plugin@^6.12.0
    
    # Formatting
    npm install -D prettier@^3.0.3
    
    # Additional dev tools
    npm install -D @vitejs/plugin-react@^4.1.1 vite@^4.5.0
    
    log "✅ Development Dependencies installiert"
}

# Initialize Tailwind CSS
init_tailwind() {
    log "🎨 Initialisiere Tailwind CSS..."
    
    # Create Tailwind config if it doesn't exist
    if [ ! -f "tailwind.config.js" ]; then
        npx tailwindcss init -p
        
        # Update Tailwind configuration
        cat > tailwind.config.js << 'EOF'
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html",
    "./src/components/**/*.{js,jsx,ts,tsx}",
    "./src/pages/**/*.{js,jsx,ts,tsx}"
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
        },
        secondary: {
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#cbd5e1',
          400: '#94a3b8',
          500: '#64748b',
          600: '#475569',
          700: '#334155',
          800: '#1e293b',
          900: '#0f172a',
        },
        success: {
          50: '#ecfdf5',
          100: '#d1fae5',
          200: '#a7f3d0',
          300: '#6ee7b7',
          400: '#34d399',
          500: '#10b981',
          600: '#059669',
          700: '#047857',
          800: '#065f46',
          900: '#064e3b',
        },
        warning: {
          50: '#fffbeb',
          100: '#fef3c7',
          200: '#fde68a',
          300: '#fcd34d',
          400: '#fbbf24',
          500: '#f59e0b',
          600: '#d97706',
          700: '#b45309',
          800: '#92400e',
          900: '#78350f',
        },
        error: {
          50: '#fef2f2',
          100: '#fee2e2',
          200: '#fecaca',
          300: '#fca5a5',
          400: '#f87171',
          500: '#ef4444',
          600: '#dc2626',
          700: '#b91c1c',
          800: '#991b1b',
          900: '#7f1d1d',
        },
        accent: {
          50: '#faf5ff',
          100: '#f3e8ff',
          200: '#e9d5ff',
          300: '#d8b4fe',
          400: '#c084fc',
          500: '#a855f7',
          600: '#9333ea',
          700: '#7c3aed',
          800: '#6b21a8',
          900: '#581c87',
        },
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'Noto Sans', 'sans-serif', 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'Liberation Mono', 'Courier New', 'monospace'],
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-in': 'slideIn 0.3s ease-out',
        'bounce-in': 'bounceIn 0.6s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideIn: {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(0)' },
        },
        bounceIn: {
          '0%': { transform: 'scale(0.3)', opacity: '0' },
          '50%': { transform: 'scale(1.05)' },
          '70%': { transform: 'scale(0.9)' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
  ],
}
EOF
        
        # Create PostCSS config
        cat > postcss.config.js << 'EOF'
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
EOF
        
        log "✅ Tailwind CSS konfiguriert"
    else
        log "✅ Tailwind CSS existiert bereits"
    fi
}

# Create main App component
create_app_component() {
    log "⚛️ Erstelle Haupt App Komponente..."
    
    cat > ~/football-ai-system/frontend/src/App.tsx << 'EOF'
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './styles/globals.css';

// Layout Components
import Layout from './components/layout/Layout';

// Pages
import Dashboard from './pages/Dashboard';
import Predictions from './pages/Predictions';
import Dutching from './pages/Dutching';
import Matches from './pages/Matches';
import Analytics from './pages/Analytics';
import Settings from './pages/Settings';

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      staleTime: 1000 * 60 * 5, // 5 minutes
      retry: 1,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="App min-h-screen bg-gray-900 text-white">
          <Layout>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/predictions" element={<Predictions />} />
              <Route path="/dutching" element={<Dutching />} />
              <Route path="/matches" element={<Matches />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </Layout>
        </div>
      </Router>
      
      <ToastContainer
        position="top-right"
        autoClose={5000}
        hideProgressBar={false}
        newestOnTop={false}
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
        theme="dark"
      />
      
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
}

export default App;
EOF

    log "✅ App Komponente erstellt"
}

# Create main CSS file
create_main_css() {
    log "🎨 Erstelle Haupt CSS Datei..."
    
    mkdir -p ~/football-ai-system/frontend/src/styles
    
    cat > ~/football-ai-system/frontend/src/styles/globals.css << 'EOF'
@tailwind base;
@tailwind components;
@tailwind utilities;

/* Custom CSS Variables */
:root {
  --primary-50: #eff6ff;
  --primary-100: #dbeafe;
  --primary-200: #bfdbfe;
  --primary-300: #93c5fd;
  --primary-400: #60a5fa;
  --primary-500: #3b82f6;
  --primary-600: #2563eb;
  --primary-700: #1d4ed8;
  --primary-800: #1e40af;
  --primary-900: #1e3a8a;

  --secondary-50: #f8fafc;
  --secondary-100: #f1f5f9;
  --secondary-200: #e2e8f0;
  --secondary-300: #cbd5e1;
  --secondary-400: #94a3b8;
  --secondary-500: #64748b;
  --secondary-600: #475569;
  --secondary-700: #334155;
  --secondary-800: #1e293b;
  --secondary-900: #0f172a;

  --success-50: #ecfdf5;
  --success-100: #d1fae5;
  --success-200: #a7f3d0;
  --success-300: #6ee7b7;
  --success-400: #34d399;
  --success-500: #10b981;
  --success-600: #059669;
  --success-700: #047857;
  --success-800: #065f46;
  --success-900: #064e3b;

  --warning-50: #fffbeb;
  --warning-100: #fef3c7;
  --warning-200: #fde68a;
  --warning-300: #fcd34d;
  --warning-400: #fbbf24;
  --warning-500: #f59e0b;
  --warning-600: #d97706;
  --warning-700: #b45309;
  --warning-800: #92400e;
  --warning-900: #78350f;

  --error-50: #fef2f2;
  --error-100: #fee2e2;
  --error-200: #fecaca;
  --error-300: #fca5a5;
  --error-400: #f87171;
  --error-500: #ef4444;
  --error-600: #dc2626;
  --error-700: #b91c1c;
  --error-800: #991b1b;
  --error-900: #7f1d1d;

  --accent-50: #faf5ff;
  --accent-100: #f3e8ff;
  --accent-200: #e9d5ff;
  --accent-300: #d8b4fe;
  --accent-400: #c084fc;
  --accent-500: #a855f7;
  --accent-600: #9333ea;
  --accent-700: #7c3aed;
  --accent-800: #6b21a8;
  --accent-900: #581c87;
}

/* Dark mode support */
.dark {
  --bg-primary: #0f172a;
  --bg-secondary: #1e293b;
  --bg-card: #334155;
  --text-primary: #f8fafc;
  --text-secondary: #cbd5e1;
  --text-muted: #94a3b8;
  --border: #475569;
}

.light {
  --bg-primary: #ffffff;
  --bg-secondary: #f8fafc;
  --bg-card: #ffffff;
  --text-primary: #0f172a;
  --text-secondary: #334155;
  --text-muted: #64748b;
  --border: #e2e8f0;
}

/* Base styles */
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  background-color: var(--bg-primary);
  color: var(--text-primary);
  transition: all 0.3s ease;
}

/* Custom scrollbar */
::-webkit-scrollbar {
  width: 8px;
}

::-webkit-scrollbar-track {
  background: var(--bg-secondary);
}

::-webkit-scrollbar-thumb {
  background: var(--border);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: var(--text-muted);
}

/* Gradient text */
.gradient-text {
  background: linear-gradient(135deg, var(--primary-500), var(--accent-500));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

/* Glass morphism */
.glass {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
}

/* Card styles */
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
}

.card:hover {
  transform: translateY(-2px);
  box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
}

/* Button styles */
.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 12px 24px;
  border-radius: 8px;
  font-weight: 500;
  transition: all 0.2s ease;
  cursor: pointer;
  border: none;
  text-decoration: none;
}

.btn-primary {
  background: linear-gradient(135deg, var(--primary-500), var(--primary-600));
  color: white;
}

.btn-primary:hover {
  background: linear-gradient(135deg, var(--primary-600), var(--primary-700));
  transform: translateY(-1px);
}

.btn-secondary {
  background: var(--bg-secondary);
  color: var(--text-primary);
  border: 1px solid var(--border);
}

.btn-secondary:hover {
  background: var(--bg-card);
  border-color: var(--primary-500);
}

/* Loading spinner */
.spinner {
  display: inline-block;
  width: 20px;
  height: 20px;
  border: 3px solid rgba(255, 255, 255, 0.3);
  border-radius: 50%;
  border-top-color: var(--primary-500);
  animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* Animations */
.fade-in {
  animation: fadeIn 0.5s ease-in-out;
}

.slide-in {
  animation: slideIn 0.3s ease-out;
}

.bounce-in {
  animation: bounceIn 0.6s ease-out;
}

/* Responsive utilities */
@media (max-width: 768px) {
  .card {
    padding: 16px;
  }
  
  .btn {
    padding: 10px 20px;
    font-size: 14px;
  }
}

/* Print styles */
@media print {
  .no-print {
    display: none !important;
  }
}

/* Focus styles for accessibility */
:focus-visible {
  outline: 2px solid var(--primary-500);
  outline-offset: 2px;
}

/* High contrast mode support */
@media (prefers-contrast: high) {
  .card {
    border-width: 2px;
  }
  
  .btn {
    border-width: 2px;
  }
}

/* Reduced motion support */
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
EOF

    log "✅ Haupt CSS Datei erstellt"
}

# Create layout components
create_layout_components() {
    log "🏗️ Erstelle Layout Komponenten..."
    
    mkdir -p ~/football-ai-system/frontend/src/components/layout
    
    # Layout component
    cat > ~/football-ai-system/frontend/src/components/layout/Layout.tsx << 'EOF'
import React from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import Header from './Header';

const Layout: React.FC<{ children?: React.ReactNode }> = ({ children }) => {
  return (
    <div className="min-h-screen bg-gray-900 text-white flex">
      {/* Sidebar */}
      <Sidebar />
      
      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <Header />
        
        {/* Page Content */}
        <main className="flex-1 p-6 overflow-auto">
          {children || <Outlet />}
        </main>
      </div>
    </div>
  );
};

export default Layout;
EOF

    # Sidebar component
    cat > ~/football-ai-system/frontend/src/components/layout/Sidebar.tsx << 'EOF'
import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  HomeIcon, 
  ChartBarIcon, 
  CalculatorIcon, 
  TrophyIcon, 
  ChartPieIcon, 
  CogIcon,
  MenuIcon,
  XIcon
} from '@heroicons/react/24/outline';

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: HomeIcon },
  { name: 'Predictions', href: '/predictions', icon: ChartBarIcon },
  { name: 'Dutching', href: '/dutching', icon: CalculatorIcon },
  { name: 'Matches', href: '/matches', icon: TrophyIcon },
  { name: 'Analytics', href: '/analytics', icon: ChartPieIcon },
  { name: 'Settings', href: '/settings', icon: CogIcon },
];

const Sidebar: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const location = useLocation();

  return (
    <>
      {/* Mobile menu button */}
      <div className="lg:hidden fixed top-4 left-4 z-50">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="p-2 rounded-md bg-gray-800 text-white hover:bg-gray-700"
        >
          {isOpen ? <XIcon className="h-6 w-6" /> : <MenuIcon className="h-6 w-6" />}
        </button>
      </div>

      {/* Sidebar */}
      <div className={`${isOpen ? 'translate-x-0' : '-translate-x-full'} lg:translate-x-0 fixed lg:relative inset-y-0 left-0 z-40 w-64 bg-gray-800 border-r border-gray-700 transition-transform duration-300 ease-in-out`}>
        <div className="flex flex-col h-full">
          {/* Logo */}
          <div className="flex items-center justify-center h-16 px-4 border-b border-gray-700">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">AI</span>
              </div>
              <span className="text-xl font-bold gradient-text">Football AI</span>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-4 py-6 space-y-2">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href;
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  onClick={() => setIsOpen(false)}
                  className={`flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors duration-200 ${
                    isActive
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                  }`}
                >
                  <item.icon className="h-5 w-5" />
                  <span className="font-medium">{item.name}</span>
                </Link>
              );
            })}
          </nav>

          {/* Footer */}
          <div className="p-4 border-t border-gray-700">
            <div className="text-sm text-gray-400 text-center">
              <p>© 2025 Football AI System</p>
              <p>Version 1.0.0</p>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black bg-opacity-50 z-30"
          onClick={() => setIsOpen(false)}
        />
      )}
    </>
  );
};

export default Sidebar;
EOF

    # Header component
    cat > ~/football-ai-system/frontend/src/components/layout/Header.tsx << 'EOF'
import React from 'react';
import { BellIcon, UserIcon } from '@heroicons/react/24/outline';

const Header: React.FC = () => {
  return (
    <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <h1 className="text-2xl font-bold gradient-text">Dashboard</h1>
          <div className="hidden md:flex items-center space-x-2 text-sm text-gray-400">
            <span>Last updated:</span>
            <span className="text-white">{new Date().toLocaleTimeString()}</span>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          {/* System Status */}
          <div className="hidden md:flex items-center space-x-2">
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span className="text-xs text-gray-400">API</span>
            </div>
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span className="text-xs text-gray-400">Models</span>
            </div>
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span className="text-xs text-gray-400">Data</span>
            </div>
          </div>

          {/* Notifications */}
          <button className="relative p-2 text-gray-400 hover:text-white transition-colors">
            <BellIcon className="h-6 w-6" />
            <span className="absolute top-0 right-0 w-2 h-2 bg-red-500 rounded-full"></span>
          </button>

          {/* User Profile */}
          <button className="flex items-center space-x-2 p-2 text-gray-400 hover:text-white transition-colors">
            <UserIcon className="h-6 w-6" />
            <span className="hidden md:block text-sm">Admin</span>
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header;
EOF

    log "✅ Layout Komponenten erstellt"
}

# Create utility functions
create_utility_functions() {
    log "🔧 Erstelle Utility Functions..."
    
    mkdir -p ~/football-ai-system/frontend/src/utils
    
    # API utilities
    cat > ~/football-ai-system/frontend/src/utils/api.ts << 'EOF'
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Create axios instance
export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('authToken');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// API endpoints
export const apiEndpoints = {
  // Predictions
  predictions: {
    list: () => api.get('/api/v1/predictions'),
    create: (data: any) => api.post('/api/v1/predictions', data),
    get: (id: string) => api.get(`/api/v1/predictions/${id}`),
    update: (id: string, data: any) => api.put(`/api/v1/predictions/${id}`, data),
    delete: (id: string) => api.delete(`/api/v1/predictions/${id}`),
  },

  // Matches
  matches: {
    list: (params?: any) => api.get('/api/v1/matches', { params }),
    get: (id: string) => api.get(`/api/v1/matches/${id}`),
    today: () => api.get('/api/v1/matches/today'),
    upcoming: () => api.get('/api/v1/matches/upcoming'),
  },

  // Dutching
  dutching: {
    calculate: (data: any) => api.post('/api/v1/dutching/calculate', data),
    strategies: () => api.get('/api/v1/dutching/strategies'),
    execute: (data: any) => api.post('/api/v1/dutching/execute', data),
  },

  // Analytics
  analytics: {
    dashboard: () => api.get('/api/v1/analytics/dashboard'),
    performance: () => api.get('/api/v1/analytics/performance'),
    predictions: (params?: any) => api.get('/api/v1/analytics/predictions', { params }),
  },

  // System
  system: {
    health: () => api.get('/api/v1/health'),
    status: () => api.get('/api/v1/status'),
    metrics: () => api.get('/api/v1/metrics'),
  },
};

export default api;
EOF

    # Utility functions
    cat > ~/football-ai-system/frontend/src/utils/helpers.ts << 'EOF'
// Date formatting
export const formatDate = (date: string | Date, format: string = 'MMM dd, yyyy') => {
  const d = typeof date === 'string' ? new Date(date) : date;
  return d.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
};

export const formatTime = (date: string | Date) => {
  const d = typeof date === 'string' ? new Date(date) : date;
  return d.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
  });
};

// Number formatting
export const formatNumber = (num: number, decimals: number = 2) => {
  return num.toFixed(decimals);
};

export const formatCurrency = (amount: number, currency: string = 'USD') => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
  }).format(amount);
};

// Percentage formatting
export const formatPercentage = (value: number, decimals: number = 1) => {
  return `${value.toFixed(decimals)}%`;
};

// Odds formatting
export const formatOdds = (odds: number, format: 'decimal' | 'fractional' | 'american' = 'decimal') => {
  switch (format) {
    case 'decimal':
      return odds.toFixed(2);
    case 'fractional':
      // Convert decimal to fractional
      const fractional = decimalToFractional(odds);
      return `${fractional.numerator}/${fractional.denominator}`;
    case 'american':
      // Convert decimal to american
      if (odds >= 2.0) {
        return `+${Math.round((odds - 1) * 100)}`;
      } else {
        return `-${Math.round(100 / (odds - 1))}`;
      }
    default:
      return odds.toFixed(2);
  }
};

// Convert decimal odds to fractional
const decimalToFractional = (decimal: number) => {
  const reduced = reduceFraction(decimal - 1, 1);
  return {
    numerator: reduced.numerator,
    denominator: reduced.denominator,
  };
};

// Reduce fraction to simplest form
const reduceFraction = (numerator: number, denominator: number) => {
  const gcd = (a: number, b: number): number => {
    return b === 0 ? a : gcd(b, a % b);
  };
  const divisor = gcd(numerator, denominator);
  return {
    numerator: numerator / divisor,
    denominator: denominator / divisor,
  };
};

// Color utilities
export const getStatusColor = (status: string) => {
  const colors: { [key: string]: string } = {
    active: 'text-green-500',
    inactive: 'text-red-500',
    pending: 'text-yellow-500',
    completed: 'text-blue-500',
    cancelled: 'text-gray-500',
  };
  return colors[status.toLowerCase()] || 'text-gray-500';
};

export const getConfidenceColor = (confidence: number) => {
  if (confidence >= 0.8) return 'text-green-500';
  if (confidence >= 0.6) return 'text-yellow-500';
  return 'text-red-500';
};

// Local storage utilities
export const storage = {
  get: (key: string) => {
    try {
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : null;
    } catch (error) {
      console.error('Error reading from localStorage:', error);
      return null;
    }
  },

  set: (key: string, value: any) => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.error('Error writing to localStorage:', error);
    }
  },

  remove: (key: string) => {
    try {
      localStorage.removeItem(key);
    } catch (error) {
      console.error('Error removing from localStorage:', error);
    }
  },

  clear: () => {
    try {
      localStorage.clear();
    } catch (error) {
      console.error('Error clearing localStorage:', error);
    }
  },
};

// Debounce utility
export const debounce = <T extends (...args: any[]) => any>(
  func: T,
  wait: number
): ((...args: Parameters<T>) => void) => {
  let timeout: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
};

// Throttle utility
export const throttle = <T extends (...args: any[]) => any>(
  func: T,
  limit: number
): ((...args: Parameters<T>) => void) => {
  let inThrottle: boolean;
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
};

// Validation utilities
export const validators = {
  email: (email: string): boolean => {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
  },

  phone: (phone: string): boolean => {
    const re = /^\+?[\d\s-()]+$/;
    return re.test(phone);
  },

  url: (url: string): boolean => {
    try {
      new URL(url);
      return true;
    } catch {
      return false;
    }
  },
};

// Performance utilities
export const measurePerformance = (name: string, fn: () => void) => {
  const start = performance.now();
  fn();
  const end = performance.now();
  console.log(`${name} took ${end - start} milliseconds`);
};

// Error handling
export const handleError = (error: any): string => {
  if (error.response?.data?.message) {
    return error.response.data.message;
  }
  if (error.message) {
    return error.message;
  }
  return 'An unexpected error occurred';
};

export const logError = (error: any, context?: string) => {
  console.error(`Error${context ? ` in ${context}` : ''}:`, error);
  // Here you could send errors to a logging service
};

// Array utilities
export const uniqueBy = <T, K extends keyof T>(array: T[], key: K): T[] => {
  const seen = new Set<T[K]>();
  return array.filter((item) => {
    const value = item[key];
    if (seen.has(value)) {
      return false;
    }
    seen.add(value);
    return true;
  });
};

export const groupBy = <T, K extends keyof T>(array: T[], key: K): Record<string, T[]> => {
  return array.reduce((groups, item) => {
    const group = String(item[key]);
    groups[group] = groups[group] || [];
    groups[group].push(item);
    return groups;
  }, {} as Record<string, T[]>);
};

// Object utilities
export const pick = <T, K extends keyof T>(obj: T, keys: K[]): Pick<T, K> => {
  return keys.reduce((result, key) => {
    if (key in obj) {
      result[key] = obj[key];
    }
    return result;
  }, {} as Pick<T, K>);
};

export const omit = <T, K extends keyof T>(obj: T, keys: K[]): Omit<T, K> => {
  const result = { ...obj };
  keys.forEach((key) => {
    delete result[key];
  });
  return result;
};

// Math utilities
export const clamp = (num: number, min: number, max: number): number => {
  return Math.min(Math.max(num, min), max);
};

export const lerp = (start: number, end: number, t: number): number => {
  return start * (1 - t) + end * t;
};

export const mapRange = (
  value: number,
  inMin: number,
  inMax: number,
  outMin: number,
  outMax: number
): number => {
  return ((value - inMin) * (outMax - outMin)) / (inMax - inMin) + outMin;
};

// Export all utilities
export default {
  formatDate,
  formatTime,
  formatNumber,
  formatCurrency,
  formatPercentage,
  formatOdds,
  getStatusColor,
  getConfidenceColor,
  storage,
  debounce,
  throttle,
  validators,
  measurePerformance,
  handleError,
  logError,
  uniqueBy,
  groupBy,
  pick,
  omit,
  clamp,
  lerp,
  mapRange,
};
EOF

    log "✅ Utility Functions erstellt"
}

# Create sample pages
create_sample_pages() {
    log "📄 Erstelle Sample Pages..."
    
    # Dashboard page
    cat > ~/football-ai-system/frontend/src/pages/Dashboard.tsx << 'EOF'
import React from 'react';
import { 
  ChartBarIcon, 
  TrophyIcon, 
  CurrencyDollarIcon, 
  ArrowTrendingUpIcon 
} from '@heroicons/react/24/outline';

const Dashboard: React.FC = () => {
  const stats = [
    { name: 'Total Predictions', value: '1,234', icon: ChartBarIcon, change: '+12%', changeType: 'increase' },
    { name: 'Success Rate', value: '78.5%', icon: TrophyIcon, change: '+2.3%', changeType: 'increase' },
    { name: 'Total Profit', value: '$12,456', icon: CurrencyDollarIcon, change: '+5.7%', changeType: 'increase' },
    { name: 'ROI', value: '15.2%', icon: ArrowTrendingUpIcon, change: '+1.1%', changeType: 'increase' },
  ];

  const recentPredictions = [
    { match: 'Manchester United vs Liverpool', prediction: 'Over 2.5 Goals', confidence: 85, odds: 1.85, result: 'Win' },
    { match: 'Arsenal vs Chelsea', prediction: 'Arsenal Win', confidence: 72, odds: 2.10, result: 'Pending' },
    { match: 'Real Madrid vs Barcelona', prediction: 'Both Teams to Score', confidence: 91, odds: 1.65, result: 'Win' },
    { match: 'Bayern Munich vs Dortmund', prediction: 'Under 3.5 Goals', confidence: 68, odds: 1.95, result: 'Loss' },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold gradient-text">Dashboard</h1>
        <div className="flex items-center space-x-4">
          <button className="btn btn-primary">
            Generate Predictions
          </button>
          <button className="btn btn-secondary">
            Refresh Data
          </button>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat) => {
          const Icon = stat.icon;
          return (
            <div key={stat.name} className="card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-400">{stat.name}</p>
                  <p className="text-2xl font-bold text-white mt-1">{stat.value}</p>
                </div>
                <div className="p-3 bg-gray-700 rounded-lg">
                  <Icon className="h-6 w-6 text-blue-500" />
                </div>
              </div>
              <div className="mt-4 flex items-center">
                <span className={`text-sm font-medium ${
                  stat.changeType === 'increase' ? 'text-green-400' : 'text-red-400'
                }`}>
                  {stat.change}
                </span>
                <span className="text-sm text-gray-400 ml-2">vs last month</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-lg font-semibold text-white mb-4">Prediction Accuracy</h3>
          <div className="h-64 bg-gray-700 rounded-lg flex items-center justify-center">
            <p className="text-gray-400">Chart Component Coming Soon</p>
          </div>
        </div>

        <div className="card">
          <h3 className="text-lg font-semibold text-white mb-4">Profit Over Time</h3>
          <div className="h-64 bg-gray-700 rounded-lg flex items-center justify-center">
            <p className="text-gray-400">Chart Component Coming Soon</p>
          </div>
        </div>
      </div>

      {/* Recent Predictions */}
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-white">Recent Predictions</h3>
          <button className="text-blue-500 hover:text-blue-400 text-sm font-medium">
            View All
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Match</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Prediction</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Confidence</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Odds</th>
                <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Result</th>
              </tr>
            </thead>
            <tbody>
              {recentPredictions.map((prediction, index) => (
                <tr key={index} className="border-b border-gray-700 hover:bg-gray-800">
                  <td className="py-3 px-4 text-sm text-white">{prediction.match}</td>
                  <td className="py-3 px-4 text-sm text-gray-300">{prediction.prediction}</td>
                  <td className="py-3 px-4 text-sm">
                    <span className={`font-medium ${getConfidenceColor(prediction.confidence)}`}>
                      {prediction.confidence}%
                    </span>
                  </td>
                  <td className="py-3 px-4 text-sm text-gray-300">{prediction.odds}</td>
                  <td className="py-3 px-4 text-sm">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      prediction.result === 'Win' ? 'bg-green-100 text-green-800' :
                      prediction.result === 'Loss' ? 'bg-red-100 text-red-800' :
                      'bg-yellow-100 text-yellow-800'
                    }`}>
                      {prediction.result}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

const getConfidenceColor = (confidence: number) => {
  if (confidence >= 80) return 'text-green-500';
  if (confidence >= 60) return 'text-yellow-500';
  return 'text-red-500';
};

export default Dashboard;
EOF

    # Other pages
    cat > ~/football-ai-system/frontend/src/pages/Predictions.tsx << 'EOF'
import React from 'react';

const Predictions: React.FC = () => {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold gradient-text">Predictions</h1>
      <div className="card">
        <h3 className="text-lg font-semibold text-white mb-4">AI Prediction Engine</h3>
        <p className="text-gray-400">Advanced machine learning models analyzing football matches for accurate predictions.</p>
      </div>
    </div>
  );
};

export default Predictions;
EOF

    cat > ~/football-ai-system/frontend/src/pages/Dutching.tsx << 'EOF'
import React from 'react';

const Dutching: React.FC = () => {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold gradient-text">Dutching System</h1>
      <div className="card">
        <h3 className="text-lg font-semibold text-white mb-4">Smart Betting Strategy</h3>
        <p className="text-gray-400">Automated dutching calculations for optimal risk distribution and profit maximization.</p>
      </div>
    </div>
  );
};

export default Dutching;
EOF

    cat > ~/football-ai-system/frontend/src/pages/Matches.tsx << 'EOF'
import React from 'react';

const Matches: React.FC = () => {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold gradient-text">Matches</h1>
      <div className="card">
        <h3 className="text-lg font-semibold text-white mb-4">Live Match Data</h3>
        <p className="text-gray-400">Real-time match information, statistics, and analysis from multiple leagues.</p>
      </div>
    </div>
  );
};

export default Matches;
EOF

    cat > ~/football-ai-system/frontend/src/pages/Analytics.tsx << 'EOF'
import React from 'react';

const Analytics: React.FC = () => {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold gradient-text">Analytics</h1>
      <div className="card">
        <h3 className="text-lg font-semibold text-white mb-4">Performance Analytics</h3>
        <p className="text-gray-400">Detailed analytics, performance metrics, and insights into prediction accuracy.</p>
      </div>
    </div>
  );
};

export default Analytics;
EOF

    cat > ~/football-ai-system/frontend/src/pages/Settings.tsx << 'EOF'
import React from 'react';

const Settings: React.FC = () => {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold gradient-text">Settings</h1>
      <div className="card">
        <h3 className="text-lg font-semibold text-white mb-4">System Configuration</h3>
        <p className="text-gray-400">Configure system preferences, API keys, and model parameters.</p>
      </div>
    </div>
  );
};

export default Settings;
EOF

    log "✅ Sample Pages erstellt"
}

# Create additional configuration files
create_additional_configs() {
    log "⚙️ Erstelle zusätzliche Konfigurationsdateien..."
    
    # TypeScript config
    cat > ~/football-ai-system/frontend/tsconfig.json << 'EOF'
{
  "compilerOptions": {
    "target": "es5",
    "lib": [
      "dom",
      "dom.iterable",
      "es6"
    ],
    "allowJs": true,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "noFallthroughCasesInSwitch": true,
    "module": "esnext",
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "baseUrl": "src",
    "paths": {
      "@/*": ["*"],
      "@components/*": ["components/*"],
      "@pages/*": ["pages/*"],
      "@utils/*": ["utils/*"],
      "@services/*": ["services/*"],
      "@hooks/*": ["hooks/*"],
      "@types/*": ["types/*"],
      "@styles/*": ["styles/*"]
    }
  },
  "include": [
    "src"
  ]
}
EOF

    # ESLint config
    cat > ~/football-ai-system/frontend/.eslintrc.json << 'EOF'
{
  "extends": [
    "react-app",
    "react-app/jest"
  ],
  "rules": {
    "no-unused-vars": "error",
    "no-console": "warn",
    "prefer-const": "error",
    "no-var": "error",
    "react/prop-types": "off",
    "react/react-in-jsx-scope": "off",
    "@typescript-eslint/no-unused-vars": "error",
    "@typescript-eslint/explicit-function-return-type": "off",
    "@typescript-eslint/explicit-module-boundary-types": "off"
  },
  "env": {
    "browser": true,
    "es6": true,
    "node": true
  }
}
EOF

    # Prettier config
    cat > ~/football-ai-system/frontend/.prettierrc << 'EOF'
{
  "semi": true,
  "trailingComma": "es5",
  "singleQuote": true,
  "printWidth": 80,
  "tabWidth": 2,
  "useTabs": false,
  "bracketSpacing": true,
  "arrowParens": "avoid",
  "endOfLine": "lf"
}
EOF

    log "✅ Zusätzliche Konfigurationsdateien erstellt"
}

# Create Docker configuration for frontend
create_docker_config() {
    log "🐳 Erstelle Frontend Docker Konfiguration..."
    
    cat > ~/football-ai-system/frontend/Dockerfile << 'EOF'
# Build stage
FROM node:18-alpine as build

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Build the app
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built app from build stage
COPY --from=build /app/build /usr/share/nginx/html

# Copy nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
EOF

    cat > ~/football-ai-system/frontend/nginx.conf << 'EOF'
server {
    listen 80;
    server_name localhost;

    root /usr/share/nginx/html;
    index index.html index.htm;

    location / {
        try_files $uri $uri/ /index.html;
    }

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied expired no-cache no-store private must-revalidate;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
EOF

    log "✅ Frontend Docker Konfiguration erstellt"
}

# Main execution
main() {
    check_user
    setup_environment
    
    log "🚀 Starte Frontend Setup (React 18 + TypeScript)"
    
    # Install dependencies
    init_react_app
    install_core_dependencies
    install_ui_dependencies
    install_chart_dependencies
    install_dev_dependencies
    
    # Configure
    init_tailwind
    create_app_component
    create_main_css
    create_layout_components
    create_utility_functions
    create_sample_pages
    create_additional_configs
    create_docker_config
    
    log "✅ Frontend Setup abgeschlossen!"
    log "📦 Dependencies werden installiert..."
    
    # Install dependencies
    npm install
    
    log "✅ Alle Dependencies installiert!"
    log "🚀 Nächster Schritt: ./05-docker-setup.sh ausführen"
    log "🎯 Um Frontend zu starten: npm start"
}

# Execute main function
main "$@"