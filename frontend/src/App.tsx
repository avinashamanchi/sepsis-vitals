import { Routes, Route } from 'react-router-dom'
import { EulaGate } from './components/EulaGate'
import { Sidebar } from './components/Sidebar'
import { TopBar } from './components/TopBar'
import { BottomNav } from './components/BottomNav'
import { Dashboard } from './pages/Dashboard'
import { Patients } from './pages/Patients'
import { ScoreLab } from './pages/ScoreLab'
import { Predict } from './pages/Predict'
import { Analytics } from './pages/Analytics'
import { Alerts } from './pages/Alerts'
import { Admin } from './pages/Admin'

export default function App() {
  return (
    <EulaGate>
      <div className="min-h-screen bg-background text-text-primary font-mono">
        <Sidebar />
        <div className="lg:ml-[220px] min-h-screen flex flex-col">
          <TopBar />
          <main className="flex-1 p-4 lg:p-6 pb-20 lg:pb-6">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/patients" element={<Patients />} />
              <Route path="/scores" element={<ScoreLab />} />
              <Route path="/predict" element={<Predict />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/alerts" element={<Alerts />} />
              <Route path="/admin" element={<Admin />} />
            </Routes>
          </main>
        </div>
        <BottomNav />
      </div>
    </EulaGate>
  )
}
