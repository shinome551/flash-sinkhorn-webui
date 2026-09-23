import { useMutation, useQuery } from '@tanstack/react-query'
import { listSamples, loadSample } from '../api/client.ts'
import type { SamplePairResponse } from '../api/types.ts'
import { describeError } from '../lib/errors.ts'

/** 同梱サンプル (assets/samples) の 2 枚をワンクリックで A・B に入れる。一覧が取れなければ何も出さない。 */
export function SamplePicker({ onLoad }: { onLoad: (pair: SamplePairResponse) => void }) {
  const samples = useQuery({ queryKey: ['samples'], queryFn: ({ signal }) => listSamples(signal), retry: false })
  const load = useMutation({ mutationFn: (id: string) => loadSample(id), onSuccess: onLoad })

  if (!samples.data?.length) return null
  const error = load.error ? describeError(load.error) : null
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs" aria-label="サンプル">
      <span className="text-slate-400">サンプルで試す:</span>
      {samples.data.map((s) => (
        <button
          key={s.id}
          type="button"
          title={s.description}
          disabled={load.isPending}
          onClick={() => load.mutate(s.id)}
          className="rounded-full bg-slate-800 px-3 py-1 text-slate-200 ring-1 ring-slate-700 hover:bg-slate-700 disabled:cursor-wait disabled:opacity-60"
        >
          {load.isPending && load.variables === s.id ? '読み込み中…' : s.title}
        </button>
      ))}
      {error && (
        <span className="text-red-400" role="alert">
          {error.title}
        </span>
      )}
    </div>
  )
}
