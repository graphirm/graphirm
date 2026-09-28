import styles from '../App.module.css';

export type ChatTab = 'chat' | 'review' | 'rules' | 'graph';

const TABS: { id: ChatTab; label: string }[] = [
  { id: 'chat', label: 'Chat' },
  { id: 'review', label: 'Review' },
  { id: 'rules', label: 'Rules' },
  { id: 'graph', label: 'Graph' },
];

interface TabBarProps {
  active: ChatTab;
  onChange: (tab: ChatTab) => void;
}

export function TabBar({ active, onChange }: TabBarProps) {
  return (
    <nav className={styles.tabBar} aria-label="Sections">
      {TABS.map((tab) => (
        <button
          key={tab.id}
          type="button"
          className={active === tab.id ? styles.tabActive : styles.tab}
          aria-current={active === tab.id ? 'page' : undefined}
          onClick={() => onChange(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </nav>
  );
}
