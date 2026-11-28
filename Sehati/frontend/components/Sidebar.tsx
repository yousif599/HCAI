/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import {
  AlertTriangle,
  Check,
  Languages,
  Moon,
  PanelLeftClose,
  PanelRightClose,
  Pencil,
  PhoneCall,
  Pin,
  PinOff,
  Search,
  SquarePen,
  Stethoscope,
  Sun,
  Trash2,
  Type,
  X,
} from 'lucide-react';
import { ChatSession, Language } from '../types';
import { ShortcutTooltip } from './ShortcutTooltip';

const TAB_PILL_SPRING = {
  type: 'spring' as const,
  stiffness: 460,
  damping: 34,
  mass: 0.7,
};

interface SidebarProps {
  activeTab: 'chat' | 'emergency' | 'guide';
  setActiveTab: (tab: 'chat' | 'emergency' | 'guide') => void;
  sessions: ChatSession[];
  currentSessionId: string | null;
  onNewChat: () => void;
  onSelectSession: (id: string) => void;
  onDeleteSession: (id: string, e: React.MouseEvent) => void;
  onRenameSession: (id: string, newTitle: string) => void;
  onTogglePinSession: (id: string) => void;
  onClearAllSessions: () => void;
  lang: Language;
  toggleLang: () => void;
  openAppearanceModal: () => void;
  isDark: boolean;
  toggleTheme: () => void;
  isSidebarOpen: boolean;
  toggleSidebar: () => void;
  searchFocusSeq?: number;
  renameTrigger?: { sessionId: string; seq: number } | null;
  deleteTrigger?: { sessionId: string; seq: number } | null;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  setActiveTab,
  sessions,
  currentSessionId,
  onNewChat,
  onSelectSession,
  onDeleteSession,
  onRenameSession,
  onTogglePinSession,
  onClearAllSessions,
  lang,
  toggleLang,
  openAppearanceModal,
  isDark,
  toggleTheme,
  isSidebarOpen,
  toggleSidebar,
  searchFocusSeq = 0,
  renameTrigger = null,
  deleteTrigger = null,
}) => {
  const isAr = lang === 'ar';
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [sidebarSearchQuery, setSidebarSearchQuery] = useState('');
  const [sessionToDelete, setSessionToDelete] = useState<string | null>(null);
  const [showClearAllConfirm, setShowClearAllConfirm] = useState(false);
  const [renamingSessionId, setRenamingSessionId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState('');

  const searchInputRef = useRef<HTMLInputElement>(null);
  const renameInputRef = useRef<HTMLInputElement>(null);
  const appliedSearchSeqRef = useRef(0);
  const appliedRenameSeqRef = useRef(0);
  const appliedDeleteSeqRef = useRef(0);

  useEffect(() => {
    if (!isSidebarOpen) {
      setIsSearchOpen(false);
      setRenamingSessionId(null);
    }
  }, [isSidebarOpen]);

  useEffect(() => {
    if (!isSearchOpen) {
      setSidebarSearchQuery('');
    }
  }, [isSearchOpen]);

  useEffect(() => {
    if (searchFocusSeq > 0 && appliedSearchSeqRef.current !== searchFocusSeq) {
      appliedSearchSeqRef.current = searchFocusSeq;
      setIsSearchOpen(true);
      const timer = window.setTimeout(() => {
        searchInputRef.current?.focus();
        searchInputRef.current?.select();
      }, 60);
      return () => window.clearTimeout(timer);
    }
  }, [searchFocusSeq]);

  useEffect(() => {
    if (renameTrigger && appliedRenameSeqRef.current !== renameTrigger.seq) {
      appliedRenameSeqRef.current = renameTrigger.seq;
      const target = sessions.find((s) => s.id === renameTrigger.sessionId);
      if (target) {
        setRenamingSessionId(target.id);
        setRenameDraft(target.title);
      }
    }
  }, [renameTrigger, sessions]);

  useEffect(() => {
    if (renamingSessionId) {
      const timer = window.setTimeout(() => {
        renameInputRef.current?.focus();
        renameInputRef.current?.select();
      }, 40);
      return () => window.clearTimeout(timer);
    }
  }, [renamingSessionId]);

  useEffect(() => {
    if (deleteTrigger && appliedDeleteSeqRef.current !== deleteTrigger.seq) {
      appliedDeleteSeqRef.current = deleteTrigger.seq;
      if (sessions.some((s) => s.id === deleteTrigger.sessionId)) {
        setSessionToDelete(deleteTrigger.sessionId);
      }
    }
  }, [deleteTrigger, sessions]);

  const filteredSessions = useMemo(() => {
    const q = sidebarSearchQuery.trim().toLowerCase();
    const list = !q
      ? sessions
      : sessions.filter((s) => {
          const matchTitle = s.title.toLowerCase().includes(q);
          const matchMessage = s.messages?.some((m) => m.text.toLowerCase().includes(q));
          return matchTitle || matchMessage;
        });

    return [...list].sort((a, b) => {
      const pinDiff = Number(Boolean(b.pinned)) - Number(Boolean(a.pinned));
      if (pinDiff !== 0) return pinDiff;
      if (a.pinned && b.pinned) {
        return (b.pinnedAt ?? b.updatedAt) - (a.pinnedAt ?? a.updatedAt);
      }
      return b.updatedAt - a.updatedAt;
    });
  }, [sessions, sidebarSearchQuery]);

  const startRenamingSession = (session: ChatSession, e?: React.MouseEvent) => {
    e?.stopPropagation();
    setRenamingSessionId(session.id);
    setRenameDraft(session.title);
  };

  const commitRename = () => {
    if (!renamingSessionId) return;
    const trimmed = renameDraft.trim();
    if (trimmed) {
      onRenameSession(renamingSessionId, trimmed);
    }
    setRenamingSessionId(null);
    setRenameDraft('');
  };

  const cancelRename = () => {
    setRenamingSessionId(null);
    setRenameDraft('');
  };

  const confirmDeleteSession = () => {
    if (sessionToDelete) {
      onDeleteSession(sessionToDelete, { stopPropagation: () => {} } as React.MouseEvent);
      setSessionToDelete(null);
    }
  };

  useEffect(() => {
    if (!sessionToDelete) return;
    const onModalKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        setSessionToDelete(null);
      } else if (e.key === 'Enter') {
        e.preventDefault();
        confirmDeleteSession();
      }
    };
    window.addEventListener('keydown', onModalKeyDown);
    return () => window.removeEventListener('keydown', onModalKeyDown);
  }, [sessionToDelete]);

  const confirmClearAll = () => {
    onClearAllSessions();
    setShowClearAllConfirm(false);
  };

  const closeSidebarOnMobile = () => {
    if (typeof window !== 'undefined' && window.innerWidth < 768 && isSidebarOpen) {
      toggleSidebar();
    }
  };

  return (
    <>
      {isSidebarOpen && (
        <div
          aria-hidden="true"
          onClick={toggleSidebar}
          className="fixed inset-0 z-30 bg-slate-900/30 md:hidden"
        />
      )}

      <aside
        className={`fixed md:static inset-y-0 start-0 z-40 transition-[width,opacity,transform] duration-150 ease-out shrink-0 h-full flex flex-col bg-white dark:bg-slate-950 border-e border-slate-200/80 dark:border-slate-800/80 rounded-none overflow-hidden shadow-xl md:shadow-none ${
          isSidebarOpen
            ? 'w-72 sm:w-[18.5rem] opacity-100 translate-x-0'
            : 'w-0 opacity-0 pointer-events-none border-e-0 p-0 shadow-none'
        }`}
      >
        {/* Sidebar Header inside aside: Logo + Full Brand + Search & Collapse */}
        <div className="px-3 pt-3 pb-2 flex items-center justify-between gap-2 shrink-0">
          <div className="flex items-center gap-2 shrink-0">
            <ShortcutTooltip
              label={isAr ? 'إخفاء القائمة الجانبية' : 'Toggle sidebar'}
              shortcut="Ctrl + B"
              placement="bottom"
            >
              <img
                src="/logo.png"
                alt={isAr ? 'شعار صحتي' : 'Sehati Logo'}
                role="button"
                tabIndex={0}
                onClick={toggleSidebar}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    toggleSidebar();
                  }
                }}
                className="w-8 h-8 object-contain rounded-lg cursor-pointer hover:scale-105 active:scale-95 transition-transform shrink-0"
              />
            </ShortcutTooltip>
            <span
              className={`font-black text-lg sm:text-xl text-slate-900 dark:text-white font-['Cairo'] select-none leading-none inline-block whitespace-nowrap ${
                isAr ? 'tracking-[-0.02em] [letter-spacing:-0.02em]' : 'tracking-normal'
              }`}
              style={{
                textRendering: 'geometricPrecision',
                fontVariantLigatures: 'normal',
              }}
            >
              {isAr ? 'صحتـــــــــي' : 'Sehati'}
            </span>
          </div>

          <div className="flex items-center gap-0.5 shrink-0">
            <ShortcutTooltip
              label={isAr ? 'بحث في المحادثات' : 'Search history'}
              shortcut="Ctrl + Shift + K"
              placement="bottom"
            >
              <button
                type="button"
                onClick={() => setIsSearchOpen((prev) => !prev)}
                className={`p-1.5 rounded-lg transition-colors cursor-pointer ${
                  isSearchOpen
                    ? 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-300'
                    : 'text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800'
                }`}
                aria-label={isAr ? 'بحث في المحادثات' : 'Search history'}
              >
                <Search className="w-4 h-4" />
              </button>
            </ShortcutTooltip>

            <ShortcutTooltip
              label={isAr ? 'إخفاء القائمة الجانبية' : 'Toggle sidebar'}
              shortcut="Ctrl + B"
              placement="bottom"
            >
              <button
                type="button"
                onClick={toggleSidebar}
                className="p-1.5 rounded-lg text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors cursor-pointer"
                aria-label={isAr ? 'إخفاء القائمة الجانبية' : 'Toggle sidebar'}
              >
                {isAr ? (
                  <PanelRightClose className="w-4 h-4" />
                ) : (
                  <PanelLeftClose className="w-4 h-4" />
                )}
              </button>
            </ShortcutTooltip>
          </div>
        </div>

        {/* Collapsible Search Input */}
        <AnimatePresence initial={false}>
          {isSearchOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.18, ease: [0.22, 1, 0.36, 1] }}
              className="px-3 pt-1 pb-1 shrink-0 overflow-hidden"
            >
              <div className="relative flex items-center">
                <Search className="w-3.5 h-3.5 text-slate-400 absolute start-2.5 pointer-events-none" />
                <input
                  ref={searchInputRef}
                  type="text"
                  autoFocus
                  value={sidebarSearchQuery}
                  onChange={(e) => setSidebarSearchQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Escape') {
                      e.preventDefault();
                      setIsSearchOpen(false);
                    }
                  }}
                  placeholder={isAr ? 'ابحث في المحادثات...' : 'Search past chats...'}
                  className="w-full bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl text-xs py-1.5 ps-8 pe-7 text-slate-800 dark:text-slate-100 placeholder:text-slate-400 focus:outline-hidden focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 transition-colors"
                />
                {sidebarSearchQuery && (
                  <button
                    type="button"
                    onClick={() => setSidebarSearchQuery('')}
                    className="absolute end-2 p-0.5 rounded text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 cursor-pointer"
                    title={isAr ? 'مسح البحث' : 'Clear search'}
                  >
                    <X className="w-3 h-3" />
                  </button>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Primary Navigation Links */}
        <nav className="px-2.5 pt-2 space-y-1 shrink-0" aria-label={isAr ? 'التنقل الرئيسي' : 'Primary navigation'}>
          {(() => {
            const hasActiveSavedSession = Boolean(
              currentSessionId && sessions.some((s) => s.id === currentSessionId)
            );
            const isNewChatActive = activeTab === 'chat' && !hasActiveSavedSession;
            return (
              <ShortcutTooltip
                label={isAr ? 'دردشة جديدة' : 'New Chat'}
                shortcut="Ctrl + Shift + O"
                placement="end"
                className="block w-full"
              >
                <button
                  type="button"
                  data-tour="new-chat"
                  aria-current={isNewChatActive ? 'page' : undefined}
                  onClick={() => {
                    onNewChat();
                    closeSidebarOnMobile();
                  }}
                  className={`relative w-full px-3.5 sm:px-4 py-2.5 rounded-full text-sm font-medium flex items-center gap-3 transition-colors duration-200 active:scale-[0.98] touch-manipulation select-none whitespace-nowrap cursor-pointer ${
                    isNewChatActive
                      ? 'text-emerald-700 dark:text-emerald-300 font-semibold'
                      : 'text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800/70'
                  }`}
                >
                  {isNewChatActive && (
                    <motion.span
                      layoutId="sidebar-nav-active-pill"
                      transition={TAB_PILL_SPRING}
                      aria-hidden="true"
                      className="absolute inset-0 rounded-full bg-emerald-500/10 dark:bg-emerald-500/15 border border-emerald-500/20 dark:border-emerald-400/20"
                    />
                  )}
                  <SquarePen className="relative z-10 w-[18px] h-[18px] shrink-0" />
                  <span className="relative z-10 truncate">{isAr ? 'دردشة جديدة' : 'New Chat'}</span>
                </button>
              </ShortcutTooltip>
            );
          })()}

          <button
            type="button"
            aria-current={activeTab === 'emergency' ? 'page' : undefined}
            onClick={() => {
              setActiveTab('emergency');
              closeSidebarOnMobile();
            }}
            className={`relative w-full px-3.5 sm:px-4 py-2.5 rounded-full text-sm font-medium flex items-center gap-3 transition-colors duration-200 active:scale-[0.98] touch-manipulation select-none whitespace-nowrap cursor-pointer ${
              activeTab === 'emergency'
                ? 'text-emerald-700 dark:text-emerald-300 font-semibold'
                : 'text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800/70'
            }`}
          >
            {activeTab === 'emergency' && (
              <motion.span
                layoutId="sidebar-nav-active-pill"
                transition={TAB_PILL_SPRING}
                aria-hidden="true"
                className="absolute inset-0 rounded-full bg-emerald-500/10 dark:bg-emerald-500/15 border border-emerald-500/20 dark:border-emerald-400/20"
              />
            )}
            <PhoneCall className="relative z-10 w-[18px] h-[18px] shrink-0" />
            <span className="relative z-10 truncate">{isAr ? 'أرقام الطوارئ' : 'Emergency'}</span>
          </button>

          <button
            type="button"
            aria-current={activeTab === 'guide' ? 'page' : undefined}
            onClick={() => {
              setActiveTab('guide');
              closeSidebarOnMobile();
            }}
            className={`relative w-full px-3.5 sm:px-4 py-2.5 rounded-full text-sm font-medium flex items-center gap-3 transition-colors duration-200 active:scale-[0.98] touch-manipulation select-none whitespace-nowrap cursor-pointer ${
              activeTab === 'guide'
                ? 'text-emerald-700 dark:text-emerald-300 font-semibold'
                : 'text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800/70'
            }`}
          >
            {activeTab === 'guide' && (
              <motion.span
                layoutId="sidebar-nav-active-pill"
                transition={TAB_PILL_SPRING}
                aria-hidden="true"
                className="absolute inset-0 rounded-full bg-emerald-500/10 dark:bg-emerald-500/15 border border-emerald-500/20 dark:border-emerald-400/20"
              />
            )}
            <Stethoscope className="relative z-10 w-[18px] h-[18px] shrink-0" />
            <span className="relative z-10 truncate">{isAr ? 'دليل الطبيب' : 'Doctor Prep'}</span>
          </button>
        </nav>

        {/* Recent Chats Header */}
        <div className="px-4 pt-5 pb-2 shrink-0">
          <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
            {isAr ? 'المحادثات الأخيرة' : 'Recent Chats'}
          </span>
        </div>

        {/* Recent Chats List */}
        <div className="flex-1 min-h-0 overflow-y-auto overscroll-y-contain px-2.5 pb-3 space-y-1 no-scrollbar">
          {sessions.length === 0 ? (
            <div className="h-full flex items-center justify-center text-center px-3 text-slate-400 dark:text-slate-500 text-xs">
              <p>{isAr ? 'مفيش محادثات سابقة' : 'No previous chats'}</p>
            </div>
          ) : filteredSessions.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center p-3 text-slate-400 text-xs">
              <p>{isAr ? 'لا توجد نتائج مطابقة' : 'No matches found'}</p>
              <button
                type="button"
                onClick={() => setSidebarSearchQuery('')}
                className="mt-1 text-[0.6875rem] text-emerald-600 hover:underline cursor-pointer"
              >
                {isAr ? 'إلغاء الفلترة' : 'Clear filter'}
              </button>
            </div>
          ) : (
            filteredSessions.map((session) => {
              const isActive = activeTab === 'chat' && session.id === currentSessionId;
              const isRenaming = renamingSessionId === session.id;
              return (
                <div
                  key={session.id}
                  onClick={() => {
                    if (isRenaming) return;
                    onSelectSession(session.id);
                    closeSidebarOnMobile();
                  }}
                  className={`relative group px-2.5 py-2 rounded-xl text-xs cursor-pointer transition-colors duration-200 flex items-center justify-between gap-1.5 ${
                    isActive
                      ? 'text-emerald-900 dark:text-emerald-100 font-semibold'
                      : 'text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800/60'
                  }`}
                >
                  {isActive && (
                    <motion.span
                      layoutId="sidebar-recent-active-pill"
                      transition={TAB_PILL_SPRING}
                      aria-hidden="true"
                      className="absolute inset-0 rounded-xl bg-emerald-500/10 dark:bg-emerald-950/50 border border-emerald-500/20 dark:border-emerald-400/20"
                    />
                  )}

                  {isRenaming ? (
                    <div
                      className="relative z-10 flex items-center gap-1 flex-1 min-w-0"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <input
                        ref={renameInputRef}
                        type="text"
                        value={renameDraft}
                        onChange={(e) => setRenameDraft(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') {
                            e.preventDefault();
                            commitRename();
                          } else if (e.key === 'Escape') {
                            e.preventDefault();
                            cancelRename();
                          }
                        }}
                        aria-label={isAr ? 'اسم المحادثة الجديد' : 'New chat name'}
                        className="flex-1 min-w-0 bg-white dark:bg-slate-900 border border-emerald-500 rounded-lg px-2 py-0.5 text-xs font-medium text-slate-900 dark:text-slate-100 focus:outline-hidden focus:ring-1 focus:ring-emerald-500"
                      />
                      <button
                        type="button"
                        onClick={commitRename}
                        className="p-1 text-emerald-600 hover:bg-emerald-500/10 rounded transition-colors cursor-pointer shrink-0"
                        aria-label={isAr ? 'حفظ الاسم' : 'Save name'}
                      >
                        <Check className="w-3.5 h-3.5" />
                      </button>
                      <button
                        type="button"
                        onClick={cancelRename}
                        className="p-1 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200 rounded transition-colors cursor-pointer shrink-0"
                        aria-label={isAr ? 'إلغاء' : 'Cancel'}
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ) : (
                    <>
                      <div className="relative z-10 flex items-center gap-1.5 flex-1 min-w-0">
                        {session.pinned && (
                          <Pin className="w-3 h-3 text-emerald-600 dark:text-emerald-400 shrink-0 fill-current" />
                        )}
                        <p className="flex-1 min-w-0 truncate">{session.title}</p>
                      </div>

                      <div
                        className={`relative z-10 flex items-center gap-0.5 shrink-0 transition-opacity ${
                          isActive
                            ? 'opacity-100'
                            : 'opacity-80 sm:opacity-0 sm:group-hover:opacity-100 sm:group-focus-within:opacity-100'
                        }`}
                      >
                        <ShortcutTooltip
                          label={
                            session.pinned
                              ? isAr
                                ? 'إلغاء تثبيت المحادثة'
                                : 'Unpin chat'
                              : isAr
                              ? 'تثبيت المحادثة'
                              : 'Pin chat'
                          }
                          shortcut="P"
                          placement="top"
                        >
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              onTogglePinSession(session.id);
                            }}
                            className={`p-1 rounded transition-colors cursor-pointer ${
                              session.pinned
                                ? 'text-emerald-600 dark:text-emerald-400 hover:bg-emerald-500/10'
                                : 'text-slate-400 hover:text-emerald-600 dark:hover:text-emerald-400'
                            }`}
                            aria-label={
                              session.pinned
                                ? isAr
                                  ? 'إلغاء تثبيت المحادثة'
                                  : 'Unpin chat'
                                : isAr
                                ? 'تثبيت المحادثة'
                                : 'Pin chat'
                            }
                          >
                            {session.pinned ? (
                              <PinOff className="w-3.5 h-3.5" />
                            ) : (
                              <Pin className="w-3.5 h-3.5" />
                            )}
                          </button>
                        </ShortcutTooltip>

                        <ShortcutTooltip
                          label={isAr ? 'إعادة تسمية المحادثة' : 'Rename chat'}
                          shortcut="R"
                          placement="top"
                        >
                          <button
                            type="button"
                            onClick={(e) => startRenamingSession(session, e)}
                            className="p-1 text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 rounded transition-colors cursor-pointer"
                            aria-label={isAr ? 'إعادة تسمية المحادثة' : 'Rename chat'}
                          >
                            <Pencil className="w-3.5 h-3.5" />
                          </button>
                        </ShortcutTooltip>

                        <ShortcutTooltip
                          label={isAr ? 'حذف المحادثة' : 'Delete chat'}
                          shortcut="D"
                          placement="top"
                        >
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              setSessionToDelete(session.id);
                            }}
                            className="p-1 text-slate-400 hover:text-red-500 rounded transition-colors cursor-pointer"
                            aria-label={isAr ? 'حذف المحادثة' : 'Delete chat'}
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </ShortcutTooltip>
                      </div>
                    </>
                  )}
                </div>
              );
            })
          )}
        </div>

        {/* Bottom Clear All Action */}
        {sessions.length > 0 && (
          <div className="px-3 py-2 border-t border-slate-100 dark:border-slate-800/70 shrink-0 flex items-center justify-end">
            <button
              type="button"
              onClick={() => setShowClearAllConfirm(true)}
              className="py-1 px-2 text-slate-400 hover:text-red-600 font-semibold flex items-center gap-1 transition-colors cursor-pointer text-[0.6875rem]"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>{isAr ? 'مسح الكل' : 'Clear All'}</span>
            </button>
          </div>
        )}

        {/* Bottom Settings: Dark/Light Mode, Font & Language */}
        <div className="px-2.5 py-2.5 border-t border-slate-200/70 dark:border-slate-800 shrink-0 grid grid-cols-3 gap-1.5">
          <button
            type="button"
            onClick={toggleTheme}
            className="flex items-center justify-center gap-1 px-2 py-1.5 text-xs font-bold rounded-xl border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors cursor-pointer"
          >
            {isDark ? (
              <Sun className="w-3.5 h-3.5 text-amber-500 shrink-0" />
            ) : (
              <Moon className="w-3.5 h-3.5 text-slate-700 dark:text-slate-300 shrink-0" />
            )}
            <span className="truncate">{isDark ? (isAr ? 'نهاري' : 'Light') : (isAr ? 'ليلي' : 'Dark')}</span>
          </button>

          <button
            type="button"
            onClick={openAppearanceModal}
            className="flex items-center justify-center gap-1 px-2 py-1.5 text-xs font-bold rounded-xl border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors cursor-pointer"
          >
            <Type className="w-3.5 h-3.5 text-teal-600 shrink-0" />
            <span className="truncate">{isAr ? 'الخط' : 'Font'}</span>
          </button>

          <button
            type="button"
            onClick={toggleLang}
            className="flex items-center justify-center gap-1 px-2 py-1.5 text-xs font-bold rounded-xl border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors cursor-pointer"
          >
            <Languages className="w-3.5 h-3.5 text-teal-600 dark:text-teal-400 shrink-0" />
            <span className="truncate">{isAr ? 'English' : 'عربي'}</span>
          </button>
        </div>
      </aside>

      {/* Delete Single Session Modal */}
      {sessionToDelete && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-xs animate-in fade-in duration-150">
          <div className="bg-white dark:bg-zinc-900 border border-slate-200 dark:border-zinc-800 rounded-2xl shadow-2xl max-w-xs sm:max-w-sm w-full p-5 text-center animate-in zoom-in-95 duration-150">
            <div className="w-12 h-12 rounded-full bg-red-100 dark:bg-red-950/60 text-red-600 dark:text-red-400 mx-auto flex items-center justify-center mb-3.5">
              <Trash2 className="w-5 h-5" />
            </div>
            <h3 className="text-base font-bold text-slate-900 dark:text-white mb-5">
              {isAr ? 'هل تريد حذف المحادثة؟' : 'Delete conversation?'}
            </h3>
            <div className="flex items-center gap-2.5">
              <button
                type="button"
                onClick={confirmDeleteSession}
                className="flex-1 py-2 px-3 text-xs font-bold rounded-xl bg-red-600 hover:bg-red-700 text-white transition-colors cursor-pointer shadow-xs"
              >
                {isAr ? 'حذف' : 'Delete'}
              </button>
              <button
                type="button"
                onClick={() => setSessionToDelete(null)}
                className="flex-1 py-2 px-3 text-xs font-semibold rounded-xl border border-slate-200 dark:border-zinc-700 hover:bg-slate-100 dark:hover:bg-zinc-800 text-slate-700 dark:text-slate-300 transition-colors cursor-pointer"
              >
                {isAr ? 'إلغاء' : 'Cancel'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Clear All Sessions Modal */}
      {showClearAllConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-xs animate-in fade-in duration-150">
          <div className="bg-white dark:bg-zinc-900 border border-slate-200 dark:border-zinc-800 rounded-2xl shadow-2xl max-w-xs sm:max-w-sm w-full p-5 text-center animate-in zoom-in-95 duration-150">
            <div className="w-12 h-12 rounded-full bg-amber-100 dark:bg-amber-950/60 text-amber-600 dark:text-amber-400 mx-auto flex items-center justify-center mb-3.5">
              <AlertTriangle className="w-5 h-5" />
            </div>
            <h3 className="text-base font-bold text-slate-900 dark:text-white mb-5">
              {isAr ? 'مسح كل سجل المحادثات؟' : 'Clear all chat history?'}
            </h3>
            <div className="flex items-center gap-2.5">
              <button
                type="button"
                onClick={confirmClearAll}
                className="flex-1 py-2 px-3 text-xs font-bold rounded-xl bg-amber-600 hover:bg-amber-700 text-white transition-colors cursor-pointer shadow-xs"
              >
                {isAr ? 'مسح الكل' : 'Clear All'}
              </button>
              <button
                type="button"
                onClick={() => setShowClearAllConfirm(false)}
                className="flex-1 py-2 px-3 text-xs font-semibold rounded-xl border border-slate-200 dark:border-zinc-700 hover:bg-slate-100 dark:hover:bg-zinc-800 text-slate-700 dark:text-slate-300 transition-colors cursor-pointer"
              >
                {isAr ? 'إلغاء' : 'Cancel'}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
