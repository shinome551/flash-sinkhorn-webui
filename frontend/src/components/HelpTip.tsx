import { useId } from 'react'

/** 「?」にホバー / フォーカスすると説明を出す。キーボードでも読める (button + aria-describedby)。 */
export function HelpTip({ text }: { text: string }) {
  const id = useId()
  return (
    <span className="group relative inline-flex align-middle">
      <button
        type="button"
        aria-label="説明を表示"
        aria-describedby={id}
        className="flex h-4 w-4 cursor-help items-center justify-center rounded-full text-[10px] leading-none text-slate-400 ring-1 ring-slate-600 hover:text-slate-100 focus-visible:text-slate-100 focus-visible:outline-2 focus-visible:outline-sky-400"
      >
        ?
      </button>
      <span
        id={id}
        role="tooltip"
        className="pointer-events-none absolute hidden left-0 top-full z-30 mt-1 w-72 max-w-[calc(100vw-2rem)] rounded bg-slate-950 p-2 text-xs font-normal leading-5 text-slate-200 shadow-lg ring-1 ring-slate-600 group-focus-within:block group-hover:block"
      >
        {text}
      </span>
    </span>
  )
}
