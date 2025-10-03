# hybrid_timetable_postgres.py (vincoli estesi)
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from datetime import date, timedelta, datetime
import argparse, os, csv, sys
try:
    import psycopg  # v3
except Exception:
    import psycopg2 as psycopg  # fallback

SLOTS_ALL = ["08:00","09:00","10:00","11:00","12:00","13:00","14:30","15:30"]

@dataclass(frozen=True)
class Class:
    class_id: str
    name: str
    start_date: Optional[date] = None

@dataclass(frozen=True)
class Subject:
    subject_id: str
    name: str
    block_len_min: int = 1
    macro_preferred: bool = False

@dataclass(frozen=True)
class Teacher:
    teacher_id: str
    name: str

@dataclass
class MergeGroup:
    merge_id: str
    name: str
    members: List[str]

@dataclass
class ClassSubjectPlan:
    plan_id: str
    class_or_group_id: str
    subject_id: str
    total_hours: int

@dataclass
class Availability:
    teacher_id: str
    weekday: int
    slot_labels: Set[str]

@dataclass
class SchoolCalendar:
    school_days: List[date]

@dataclass
class Lesson:
    lesson_id: str
    class_or_group_id: str
    subject_id: str
    duration_slots: int
    split_fallback: bool

@dataclass
class Assignment:
    lesson_id: str
    date: date
    start_slot: str
    end_slot: str
    teacher_id: str
    class_or_group_id: str
    subject_id: str

def slots_between(start_label: str, k: int, slots: List[str]) -> Optional[List[str]]:
    if start_label not in slots: return None
    i = slots.index(start_label); j = i + k
    if j <= len(slots):
        block = slots[i:j]
        for a,b in zip(block[:-1], block[1:]):
            if a=="13:00" and b=="14:30": return None
        return block
    return None

def weekday_it(d: date) -> str:
    return ["Lun","Mar","Mer","Gio","Ven","Sab","Dom"][d.weekday()]

def dt_local(date_obj: date, slot_label: str) -> datetime:
    h,m = map(int, slot_label.split(":"))
    return datetime(date_obj.year, date_obj.month, date_obj.day, h, m)

class TimetableSolver:
    def __init__(self, classes, subjects, teachers, teacher_subjects, availability,
                 class_plans, merge_groups, school_calendar, use_afternoon,
                 teacher_blackouts, class_stage_periods, class_start_dates,
                 class_day_capacity, default_day_capacity=6):
        self.classes = classes; self.subjects = subjects; self.teachers = teachers
        self.teacher_subjects = teacher_subjects; self.availability = availability
        self.class_plans = class_plans; self.merge_groups = merge_groups
        self.school_calendar = school_calendar; self.use_afternoon = use_afternoon
        self.teacher_blackouts = teacher_blackouts
        self.class_stage_periods = class_stage_periods
        self.class_start_dates = class_start_dates
        self.class_day_capacity = class_day_capacity
        self.default_day_capacity = default_day_capacity
        self.slots = [s for s in SLOTS_ALL if (use_afternoon or s <= "13:00")]
        self.avail_map = {}
        for av in availability:
            self.avail_map.setdefault((av.teacher_id, av.weekday), set()).update(av.slot_labels)
        self.teacher_busy=set(); self.class_busy=set(); self.group_busy=set()
        self.group_members={f"G:{g.merge_id}":list(g.members) for g in merge_groups.values()}
        self.class_hours_count={}  # (cid,date)->ore usate
        self.initial_zero_domain=[]

    def _teacher_date_allowed(self, tid, d):
        for sd,ed in self.teacher_blackouts.get(tid, []):
            if sd <= d <= ed: return False
        return True
    def _class_date_allowed(self, cid, d):
        sd = self.class_start_dates.get(cid)
        if sd and d < sd: return False
        for a,b in self.class_stage_periods.get(cid, []):
            if a <= d <= b: return False
        return True
    def _effective_members(self, cid_or_gid):
        if cid_or_gid.startswith("C:"): return [cid_or_gid.split("C:",1)[1]]
        gid = cid_or_gid.split("G:",1)[1]; return self.group_members.get(f"G:{gid}", [])
    def _capacity_ok(self, cids, d, k):
        wd=d.weekday()
        for cid in cids:
            cap=self.class_day_capacity.get((cid,wd), self.default_day_capacity)
            used=self.class_hours_count.get((cid,d),0)
            if used+k>cap: return False
        return True

    def generate_lessons(self, allow_macro_split_fallback=True):
        lessons=[]; counter=1
        for p in self.class_plans:
            subj=self.subjects[p.subject_id]; k=max(1, subj.block_len_min)
            if subj.macro_preferred and k>1:
                n=p.total_hours//k; r=p.total_hours%k
                for _ in range(n):
                    lessons.append(Lesson(f"L{counter}", p.class_or_group_id, p.subject_id, k, allow_macro_split_fallback)); counter+=1
                for _ in range(r):
                    lessons.append(Lesson(f"L{counter}", p.class_or_group_id, p.subject_id, 1, False)); counter+=1
            else:
                for _ in range(p.total_hours):
                    lessons.append(Lesson(f"L{counter}", p.class_or_group_id, p.subject_id, 1, False)); counter+=1
        return lessons

    def candidate_assignments(self, L: Lesson):
        out=[]; subj=self.subjects[L.subject_id]; k=L.duration_slots
        teachers_for=[tid for tid,subs in self.teacher_subjects.items() if subj.subject_id in subs]
        if not teachers_for: return out
        members=self._effective_members(L.class_or_group_id)
        for d in self.school_calendar.school_days:
            if any(not self._class_date_allowed(cid,d) for cid in members): continue
            for start in self.slots:
                block=slots_between(start,k,self.slots)
                if not block: continue
                for tid in teachers_for:
                    if not self._teacher_date_allowed(tid,d): continue
                    av=self.avail_map.get((tid,d.weekday()),set())
                    if all(s in av for s in block):
                        cap_any=any(self.class_day_capacity.get((cid,d.weekday()), self.default_day_capacity)>0 for cid in members)
                        if cap_any: out.append((d,start,tid))
        return out

    def order_lessons(self, lessons, domains):
        return sorted(lessons, key=lambda l:(-l.duration_slots, len(domains.get(l.lesson_id,[])), 0 if l.class_or_group_id.startswith("G:") else 1, l.lesson_id))

    def is_conflict(self, L, d, start, tid):
        k=L.duration_slots; block=slots_between(start,k,self.slots)
        if not block: return True
        for s in block:
            if (tid,d,s) in self.teacher_busy: return True
        members=self._effective_members(L.class_or_group_id)
        if not self._capacity_ok(members,d,k): return True
        if L.class_or_group_id.startswith("G:"):
            if any((L.class_or_group_id,d,s) in self.group_busy for s in block): return True
            for cid in members:
                if any((cid,d,s) in self.class_busy for s in block): return True
        else:
            cid=members[0]
            if any((cid,d,s) in self.class_busy for s in block): return True
        return False

    def place(self, L, d, start, tid):
        k=L.duration_slots; block=slots_between(start,k,self.slots); assert block
        for s in block:
            self.teacher_busy.add((tid,d,s))
            if L.class_or_group_id.startswith("G:"):
                self.group_busy.add((L.class_or_group_id,d,s))
                for cid in self._effective_members(L.class_or_group_id):
                    self.class_busy.add((cid,d,s))
            else:
                cid=self._effective_members(L.class_or_group_id)[0]
                self.class_busy.add((cid,d,s))
        for cid in self._effective_members(L.class_or_group_id):
            self.class_hours_count[(cid,d)]=self.class_hours_count.get((cid,d),0)+k

    def unplace(self, L, d, start, tid):
        k=L.duration_slots; block=slots_between(start,k,self.slots); assert block
        for s in block:
            self.teacher_busy.discard((tid,d,s))
            if L.class_or_group_id.startswith("G:"):
                self.group_busy.discard((L.class_or_group_id,d,s))
                for cid in self._effective_members(L.class_or_group_id):
                    self.class_busy.discard((cid,d,s))
            else:
                cid=self._effective_members(L.class_or_group_id)[0]
                self.class_busy.discard((cid,d,s))
        for cid in self._effective_members(L.class_or_group_id):
            self.class_hours_count[(cid,d)]=self.class_hours_count.get((cid,d),0)-k
            if self.class_hours_count[(cid,d)]<=0: del self.class_hours_count[(cid,d)]

    def search(self, lessons, domains):
        if not lessons: return []
        L=lessons[0]
        def score(c):
            d,start,tid=c
            base={"08:00":5,"09:00":3,"10:00":1,"11:00":1,"12:00":2,"13:00":3,"14:30":6,"15:30":6}.get(start,4)
            tload=sum(1 for (tt,dd,ss) in self.teacher_busy if tt==tid and dd==d)
            return (base,tload)
        for (d,start,tid) in sorted(domains[L.lesson_id], key=score):
            if not self.is_conflict(L,d,start,tid):
                self.place(L,d,start,tid)
                tail=self.search(lessons[1:],domains)
                if tail is not None:
                    end=slots_between(start,L.duration_slots,self.slots)[-1]
                    return [Assignment(L.lesson_id,d,start,end,tid,L.class_or_group_id,L.subject_id)] + tail
                self.unplace(L,d,start,tid)
        return None

    def solve(self, allow_macro_split_fallback=True, try_split_on_fail=True):
        lessons=self.generate_lessons(allow_macro_split_fallback)
        domains={}; zero=[]
        for L in lessons:
            dom=self.candidate_assignments(L); domains[L.lesson_id]=dom
            if not dom: zero.append(L)
        self.initial_zero_domain=[L.lesson_id for L in zero]
        if zero and try_split_on_fail:
            changed=False; new=[]
            for L in lessons:
                if L in zero and L.duration_slots>1 and L.split_fallback:
                    for i in range(L.duration_slots):
                        new.append(Lesson(f"{L.lesson_id}_u{i+1}",L.class_or_group_id,L.subject_id,1,False))
                    changed=True
                else: new.append(L)
            if changed:
                lessons=new; domains={}; zero=[]
                for L in lessons:
                    dom=self.candidate_assignments(L); domains[L.lesson_id]=dom
                    if not dom: zero.append(L)
                self.initial_zero_domain=[L.lesson_id for L in zero]
        if zero: return None, {"feasible": False, "reason":"Lezioni senza candidati", "lessons": self.initial_zero_domain}
        ordered=self.order_lessons(lessons, domains)
        sol=self.search(ordered, domains)
        if sol is None:
            return None, {"feasible": False, "reason":"Backtracking fallito", "zero_domain": self.initial_zero_domain}
        sol.sort(key=lambda a:(a.date,a.start_slot,a.class_or_group_id))
        return sol, {"feasible": True, "zero_domain": self.initial_zero_domain}

def connect_pg(dsn:str):
    return psycopg.connect(dsn)

def _plans_from_db(conn):
    cur=conn.cursor()
    cur.execute("""SELECT plan_id, class_id, merge_id, subject_id, total_hours FROM class_subject_plan ORDER BY plan_id""")
    out=[]
    for pid,cid,mid,sid,th in cur.fetchall():
        if cid and not mid: cg=f"C:{cid}"
        elif mid and not cid: cg=f"G:{mid}"
        else: raise ValueError(f"Piano {pid}: specificare solo class_id O merge_id.")
        out.append(ClassSubjectPlan(str(pid), cg, sid, int(th)))
    cur.close(); return out

def load_config_from_db(conn, start_date:date, days:int|None, weeks:int|None):
    cur=conn.cursor()
    cur.execute("SELECT class_id, name, start_date FROM classes ORDER BY class_id")
    classes={r[0]: Class(r[0], r[1], r[2]) for r in cur.fetchall()}
    cur.execute("SELECT subject_id, name, COALESCE(block_len_min,1), COALESCE(macro_preferred,false) FROM subjects ORDER BY subject_id")
    subjects={r[0]: Subject(r[0], r[1], int(r[2]), bool(r[3])) for r in cur.fetchall()}
    cur.execute("SELECT teacher_id, name FROM teachers ORDER BY teacher_id")
    teachers={r[0]: Teacher(r[0], r[1]) for r in cur.fetchall()}
    cur.execute("SELECT teacher_id, subject_id FROM teacher_subjects")
    teacher_subjects={}
    for tid,sid in cur.fetchall():
        teacher_subjects.setdefault(tid,set()).add(sid)
    cur.execute("SELECT merge_id, name FROM merge_groups ORDER BY merge_id")
    merge_groups={}
    for mid,name in cur.fetchall():
        merge_groups[mid]=MergeGroup(mid,name,[])
    cur.execute("SELECT merge_id, class_id FROM merge_members")
    for mid,cid in cur.fetchall():
        merge_groups[mid].members.append(cid)
    cur.execute("SELECT teacher_id, weekday, slot_label FROM teacher_availability")
    av_map={}
    for tid,wd,lab in cur.fetchall():
        av_map.setdefault((tid,int(wd)),set()).add(lab)
    availability=[Availability(tid,wd,slots) for (tid,wd),slots in av_map.items()]
    cur.execute("SELECT dt FROM holidays")
    holidays={r[0] for r in cur.fetchall()}
    cur.execute("SELECT teacher_id, start_date, end_date FROM teacher_unavailability_periods")
    teacher_blackouts={}
    for tid,sd,ed in cur.fetchall():
        teacher_blackouts.setdefault(tid,[]).append((sd,ed))
    cur.execute("SELECT class_id, start_date, end_date FROM class_stage_periods")
    class_stage={}
    for cid,sd,ed in cur.fetchall():
        class_stage.setdefault(cid,[]).append((sd,ed))
    cur.execute("SELECT class_id, weekday, max_hours FROM class_day_capacity")
    daycap={}
    for cid,wd,mh in cur.fetchall():
        daycap[(cid,int(wd))]=int(mh)
    school_days=[]
    if weeks and weeks>0:
        d=start_date
        while d.weekday()!=0: d+=timedelta(days=1)
        end=d+timedelta(days=weeks*7-1)
        dd=d
        while dd<=end:
            if dd.weekday()<5 and dd not in holidays: school_days.append(dd)
            dd+=timedelta(days=1)
    elif days and days>0:
        dd=start_date
        while len(school_days)<days:
            if dd.weekday()<5 and dd not in holidays: school_days.append(dd)
            dd+=timedelta(days=1)
    else:
        raise ValueError("Specificare --days o --weeks.")
    cur.close()
    class_start_dates={cid: c.start_date if c.start_date else None for cid,c in classes.items()}
    return (classes,subjects,teachers,teacher_subjects,merge_groups,availability,SchoolCalendar(school_days),
            teacher_blackouts,class_stage,class_start_dates,daycap)

def export_csv(schedule, classes, subjects, teachers, merge_groups, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f, delimiter=";")
        w.writerow(["Data","Giorno","Inizio","Fine","Classe/Group","Materia","Docente"])
        for a in schedule:
            if a.class_or_group_id.startswith("C:"):
                cid=a.class_or_group_id.split("C:",1)[1]; name=classes[cid].name
            else:
                gid=a.class_or_group_id.split("G:",1)[1]; name=merge_groups[gid].name
            w.writerow([a.date.isoformat(), weekday_it(a.date), a.start_slot, a.end_slot, name, subjects[a.subject_id].name, teachers[a.teacher_id].name])

def ics_escape(t:str)->str:
    return t.replace("\\","\\\\").replace("\n","\\n").replace(",","\\,").replace(";","\\;")

def export_ics_per_class(schedule, classes, subjects, teachers, merge_groups, folder, tzid="Europe/Rome"):
    os.makedirs(folder, exist_ok=True)
    class_events={cid:[] for cid in classes.keys()}
    for a in schedule:
        if a.class_or_group_id.startswith("C:"):
            cid=a.class_or_group_id.split("C:",1)[1]; class_events[cid].append(a)
        else:
            gid=a.class_or_group_id.split("G:",1)[1]
            for cid in merge_groups[gid].members: class_events[cid].append(a)
    for cid, events in class_events.items():
        path=os.path.join(folder, f"orario_classe_{cid}.ics")
        with open(path,"w",encoding="utf-8") as f:
            f.write("BEGIN:VCALENDAR\r\nVERSION:2.0\r\nPRODID:-//HybridTimetable//IT//\r\n")
            for idx,a in enumerate(events, start=1):
                start_dt=dt_local(a.date, a.start_slot)
                all_slots=[s for s in SLOTS_ALL if s <= "13:00"]
                if a.end_slot in all_slots:
                    i=all_slots.index(a.end_slot)
                    end_dt=dt_local(a.date, all_slots[i+1]) if i+1 < len(all_slots) else dt_local(a.date,a.end_slot)+timedelta(hours=1)
                else:
                    end_dt=dt_local(a.date,a.end_slot)+timedelta(hours=1)
                fmt=lambda dt: dt.strftime("%Y%m%dT%H%M%S")
                title=f"{subjects[a.subject_id].name} - {teachers[a.teacher_id].name}"
                f.write("BEGIN:VEVENT\r\n")
                f.write(f"UID:CLASS-{cid}-{a.lesson_id}-{idx}@hybrid\r\n")
                f.write(f"DTSTART;TZID={tzid}:{fmt(start_dt)}\r\n")
                f.write(f"DTEND;TZID={tzid}:{fmt(end_dt)}\r\n")
                f.write(f"SUMMARY:{ics_escape(title)}\r\n")
                f.write("END:VEVENT\r\n")
            f.write("END:VCALENDAR\r\n")

def export_ics_per_teacher(schedule, teachers, subjects, classes, merge_groups, folder, tzid="Europe/Rome"):
    os.makedirs(folder, exist_ok=True)
    teacher_events={tid:[] for tid in teachers.keys()}
    for a in schedule: teacher_events[a.teacher_id].append(a)
    for tid, events in teacher_events.items():
        path=os.path.join(folder, f"orario_docente_{tid}.ics")
        with open(path,"w",encoding="utf-8") as f:
            f.write("BEGIN:VCALENDAR\r\nVERSION:2.0\r\nPRODID:-//HybridTimetable//IT//\r\n")
            for idx,a in enumerate(events, start=1):
                start_dt=dt_local(a.date, a.start_slot)
                all_slots=[s for s in SLOTS_ALL if s <= "13:00"]
                if a.end_slot in all_slots:
                    i=all_slots.index(a.end_slot)
                    end_dt=dt_local(a.date, all_slots[i+1]) if i+1 < len(all_slots) else dt_local(a.date,a.end_slot)+timedelta(hours=1)
                else:
                    end_dt=dt_local(a.date,a.end_slot)+timedelta(hours=1)
                fmt=lambda dt: dt.strftime("%Y%m%dT%H%M%S")
                if a.class_or_group_id.startswith("C:"):
                    cid=a.class_or_group_id.split("C:",1)[1]; entity=f"Classe {classes[cid].name}"
                else:
                    gid=a.class_or_group_id.split("G:",1)[1]; entity=f"Gruppo {merge_groups[gid].name}"
                title=f"{subjects[a.subject_id].name} - {entity}"
                f.write("BEGIN:VEVENT\r\n")
                f.write(f"UID:TEACHER-{tid}-{a.lesson_id}-{idx}@hybrid\r\n")
                f.write(f"DTSTART;TZID={tzid}:{fmt(start_dt)}\r\n")
                f.write(f"DTEND;TZID={tzid}:{fmt(end_dt)}\r\n")
                f.write(f"SUMMARY:{ics_escape(title)}\r\n")
                f.write("END:VEVENT\r\n")
            f.write("END:VCALENDAR\r\n")

def write_schedule_to_db(conn, schedule, clear_range=False):
    if not schedule: return
    dmin=min(a.date for a in schedule); dmax=max(a.date for a in schedule)
    cur=conn.cursor()
    if clear_range: cur.execute("DELETE FROM schedule WHERE date BETWEEN %s AND %s", (dmin,dmax))
    for a in schedule:
        if a.class_or_group_id.startswith("C:"):
            typ="C"; cid=a.class_or_group_id.split("C:",1)[1]; gid=None
        else:
            typ="G"; gid=a.class_or_group_id.split("G:",1)[1]; cid=None
        cur.execute("""INSERT INTO schedule (date,start_slot,end_slot,class_or_group_type,class_id,merge_id,subject_id,teacher_id)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (a.date,a.start_slot,a.end_slot,typ,cid,gid,a.subject_id,a.teacher_id))
    conn.commit(); cur.close()

def main():
    ap=argparse.ArgumentParser(description="Hybrid School Timetable (vincoli estesi)")
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--start", type=lambda s: date.fromisoformat(s))
    ap.add_argument("--days", type=int)
    ap.add_argument("--weeks", type=int)
    ap.add_argument("--out", default="./timetable_output")
    ap.add_argument("--use-afternoon", action="store_true")
    ap.add_argument("--write-db", action="store_true")
    ap.add_argument("--clear-range", action="store_true")
    args=ap.parse_args()
    if args.start is None:
        today=date.today(); args.start=today+timedelta(days=((7-today.weekday())%7))
    if args.days is None and args.weeks is None: args.weeks=1
    os.makedirs(args.out, exist_ok=True)
    with connect_pg(args.dsn) as conn:
        (classes,subjects,teachers,teacher_subjects,merge_groups,availability,school_calendar,
         teacher_blackouts,class_stage,class_start_dates,daycap)=load_config_from_db(conn,args.start,args.days,args.weeks)
        solver=TimetableSolver(classes,subjects,teachers,teacher_subjects,availability,
                               _plans_from_db(conn),merge_groups,school_calendar,args.use_afternoon,
                               teacher_blackouts,class_stage,class_start_dates,daycap,default_day_capacity=6)
        schedule, meta=solver.solve(True, True)
        if not meta.get("feasible"):
            print("⚠️ Nessuna soluzione trovata."); print(meta); sys.exit(2)
        csv_path=os.path.join(args.out,"schedule.csv")
        export_csv(schedule, classes, subjects, teachers, merge_groups, csv_path)
        export_ics_per_class(schedule, classes, subjects, teachers, merge_groups, os.path.join(args.out,"ics_classi"))
        export_ics_per_teacher(schedule, teachers, subjects, classes, merge_groups, os.path.join(args.out,"ics_docenti"))
        print("OK. CSV:", csv_path)
        if args.write_db:
            write_schedule_to_db(conn, schedule, clear_range=args.clear_range)
            print("Soluzione scritta su DB.")

if __name__=="__main__":
    main()
