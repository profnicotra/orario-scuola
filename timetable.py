# hybrid_timetable_postgres.py
# Prototype: hybrid school timetable generator with Postgres IO (DDL sample provided separately).
# - Reads classes/subjects/teachers/groups/plans/availability from Postgres
# - Builds a hybrid schedule (macro blocks for labs, unit fallback if needed)
# - Exports CSV + ICS and (optionally) writes schedule back to Postgres
#
# Usage examples:
#   python hybrid_timetable_postgres.py --dsn "postgresql://user:pass@localhost:5432/scuola_orari" --start 2025-10-06 --days 5 --out ./out --write-db --clear-range
#   python hybrid_timetable_postgres.py --dsn $DATABASE_URL --weeks 1
#
# Dependencies:
#   pip install psycopg[binary]      # preferred (psycopg v3)
#   # or
#   pip install psycopg2-binary      # fallback (psycopg2)

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set
from datetime import date, timedelta, datetime
import argparse
import os
import csv
import sys

# Try psycopg v3 then fallback to psycopg2
try:
    import psycopg  # type: ignore
    HAVE_PSYCOPG3 = True
except Exception:
    HAVE_PSYCOPG3 = False
    try:
        import psycopg2 as psycopg  # type: ignore
    except Exception as e:
        print("Errore: serve 'psycopg[binary]' o 'psycopg2-binary' (pip install).", file=sys.stderr)
        raise

# -----------------------------
# Domain
# -----------------------------

SLOTS_ALL = ["08:00","09:00","10:00","11:00","12:00","13:00","14:30","15:30"]

@dataclass(frozen=True)
class Class:
    class_id: str
    name: str

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
    class_or_group_id: str  # "C:<class_id>" or "G:<merge_id>"
    subject_id: str
    total_hours: int

@dataclass
class Availability:
    teacher_id: str
    weekday: int      # 0=Mon..6=Sun
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

# -----------------------------
# Utilities
# -----------------------------

def slots_between(start_label: str, k: int, slots: List[str]) -> Optional[List[str]]:
    if start_label not in slots:
        return None
    i = slots.index(start_label)
    j = i + k
    if j <= len(slots):
        block = slots[i:j]
        # forbid lunch break adjacency 13:00 -> 14:30
        for a, b in zip(block[:-1], block[1:]):
            if a == "13:00" and b == "14:30":
                return None
        return block
    return None

def weekday_it(d: date) -> str:
    giorni = ["Lun","Mar","Mer","Gio","Ven","Sab","Dom"]
    return giorni[d.weekday()]

def dt_local(date_obj: date, slot_label: str) -> datetime:
    hh, mm = map(int, slot_label.split(":"))
    return datetime(date_obj.year, date_obj.month, date_obj.day, hh, mm)

# -----------------------------
# Solver (backtracking)
# -----------------------------

class TimetableSolver:
    def __init__(
        self,
        classes: Dict[str, Class],
        subjects: Dict[str, Subject],
        teachers: Dict[str, Teacher],
        teacher_subjects: Dict[str, Set[str]],
        availability: List[Availability],
        class_plans: List[ClassSubjectPlan],
        merge_groups: Dict[str, MergeGroup],
        school_calendar: SchoolCalendar,
        use_afternoon: bool = False,
    ):
        self.classes = classes
        self.subjects = subjects
        self.teachers = teachers
        self.teacher_subjects = teacher_subjects
        self.availability = availability
        self.class_plans = class_plans
        self.merge_groups = merge_groups
        self.school_calendar = school_calendar
        self.use_afternoon = use_afternoon

        self.slots = [s for s in SLOTS_ALL if (use_afternoon or s <= "13:00")]

        # (teacher, weekday) -> set(slot_labels)
        self.avail_map: Dict[Tuple[str,int], Set[str]] = {}
        for av in availability:
            self.avail_map.setdefault((av.teacher_id, av.weekday), set()).update(av.slot_labels)

        # Resource occupancy during search
        self.teacher_busy: Set[Tuple[str, date, str]] = set()
        self.class_busy: Set[Tuple[str, date, str]] = set()
        self.group_busy: Set[Tuple[str, date, str]] = set()

        # Group -> members
        self.group_members: Dict[str, List[str]] = {}
        for gid, g in merge_groups.items():
            self.group_members[f"G:{gid}"] = list(g.members)

        self.initial_zero_domain: List[str] = []

    def generate_lessons(self, allow_macro_split_fallback: bool = True) -> List[Lesson]:
        lessons: List[Lesson] = []
        counter = 1
        for plan in self.class_plans:
            subj = self.subjects[plan.subject_id]
            k = max(1, subj.block_len_min)
            if subj.macro_preferred and k > 1:
                n_full = plan.total_hours // k
                rem = plan.total_hours % k
                for _ in range(n_full):
                    lessons.append(Lesson(
                        lesson_id=f"L{counter}",
                        class_or_group_id=plan.class_or_group_id,
                        subject_id=plan.subject_id,
                        duration_slots=k,
                        split_fallback=allow_macro_split_fallback
                    ))
                    counter += 1
                if rem > 0:
                    for _ in range(rem):
                        lessons.append(Lesson(
                            lesson_id=f"L{counter}",
                            class_or_group_id=plan.class_or_group_id,
                            subject_id=plan.subject_id,
                            duration_slots=1,
                            split_fallback=False
                        ))
                        counter += 1
            else:
                for _ in range(plan.total_hours):
                    lessons.append(Lesson(
                        lesson_id=f"L{counter}",
                        class_or_group_id=plan.class_or_group_id,
                        subject_id=plan.subject_id,
                        duration_slots=1,
                        split_fallback=False
                    ))
                    counter += 1
        return lessons

    def candidate_assignments(self, lesson: Lesson) -> List[Tuple[date, str, str]]:
        candidates = []
        subj = self.subjects[lesson.subject_id]
        k = lesson.duration_slots
        teachers_for_subj = [tid for tid, subs in self.teacher_subjects.items() if subj.subject_id in subs]
        if not teachers_for_subj:
            return candidates
        for d in self.school_calendar.school_days:
            wd = d.weekday()
            for start in self.slots:
                block = slots_between(start, k, self.slots)
                if not block:
                    continue
                for tid in teachers_for_subj:
                    av_slots = self.avail_map.get((tid, wd), set())
                    if all((s in av_slots) for s in block):
                        candidates.append((d, start, tid))
        return candidates

    def order_lessons(self, lessons: List[Lesson], domains: Dict[str, List[Tuple[date,str,str]]]) -> List[Lesson]:
        def key(l: Lesson):
            dom = domains.get(l.lesson_id, [])
            is_group = l.class_or_group_id.startswith("G:")
            return (-l.duration_slots, len(dom), 0 if is_group else 1, l.lesson_id)
        return sorted(lessons, key=key)

    def is_conflict(self, lesson: Lesson, d: date, start: str, tid: str) -> bool:
        k = lesson.duration_slots
        block = slots_between(start, k, self.slots)
        if not block:
            return True
        for s in block:
            if (tid, d, s) in self.teacher_busy:
                return True
        if lesson.class_or_group_id.startswith("G:"):
            if any((lesson.class_or_group_id, d, s) in self.group_busy for s in block):
                return True
            for cid in self.group_members.get(lesson.class_or_group_id, []):
                if any((cid, d, s) in self.class_busy for s in block):
                    return True
        else:
            cid = lesson.class_or_group_id.split("C:",1)[1]
            if any((cid, d, s) in self.class_busy for s in block):
                return True
        return False

    def place(self, lesson: Lesson, d: date, start: str, tid: str):
        k = lesson.duration_slots
        block = slots_between(start, k, self.slots)
        assert block
        for s in block:
            self.teacher_busy.add((tid, d, s))
            if lesson.class_or_group_id.startswith("G:"):
                self.group_busy.add((lesson.class_or_group_id, d, s))
                for cid in self.group_members.get(lesson.class_or_group_id, []):
                    self.class_busy.add((cid, d, s))
            else:
                cid = lesson.class_or_group_id.split("C:",1)[1]
                self.class_busy.add((cid, d, s))

    def unplace(self, lesson: Lesson, d: date, start: str, tid: str):
        k = lesson.duration_slots
        block = slots_between(start, k, self.slots)
        assert block
        for s in block:
            self.teacher_busy.discard((tid, d, s))
            if lesson.class_or_group_id.startswith("G:"):
                self.group_busy.discard((lesson.class_or_group_id, d, s))
                for cid in self.group_members.get(lesson.class_or_group_id, []):
                    self.class_busy.discard((cid, d, s))
            else:
                cid = lesson.class_or_group_id.split("C:",1)[1]
                self.class_busy.discard((cid, d, s))

    def search(self, lessons: List[Lesson], domains: Dict[str, List[Tuple[date,str,str]]]) -> Optional[List[Assignment]]:
        if not lessons:
            return []
        L = lessons[0]

        def candidate_score(c):
            d, start, tid = c
            base = {"08:00": 5, "09:00": 3, "10:00": 1, "11:00": 1, "12:00": 2, "13:00": 3, "14:30": 6, "15:30": 6}.get(start, 4)
            teacher_load = sum(1 for (tt, dd, ss) in self.teacher_busy if tt == tid and dd == d)
            return (base, teacher_load)

        for (d, start, tid) in sorted(domains[L.lesson_id], key=candidate_score):
            if not self.is_conflict(L, d, start, tid):
                self.place(L, d, start, tid)
                tail = lessons[1:]
                res = self.search(tail, domains)
                if res is not None:
                    block = slots_between(start, L.duration_slots, self.slots)
                    end_slot = block[-1]
                    return [Assignment(L.lesson_id, d, start, end_slot, tid, L.class_or_group_id, L.subject_id)] + res
                self.unplace(L, d, start, tid)
        return None

    def solve(self, allow_macro_split_fallback: bool = True, try_split_on_fail: bool = True):
        lessons = self.generate_lessons(allow_macro_split_fallback=allow_macro_split_fallback)
        domains: Dict[str, List[Tuple[date,str,str]]] = {}
        zero_domain_lessons = []
        for L in lessons:
            dom = self.candidate_assignments(L)
            domains[L.lesson_id] = dom
            if not dom:
                zero_domain_lessons.append(L)

        if zero_domain_lessons and try_split_on_fail:
            changed = False
            new_lessons = []
            for L in lessons:
                if L.duration_slots > 1 and L.split_fallback and L in zero_domain_lessons:
                    for i in range(L.duration_slots):
                        new_lessons.append(Lesson(
                            lesson_id=f"{L.lesson_id}_u{i+1}",
                            class_or_group_id=L.class_or_group_id,
                            subject_id=L.subject_id,
                            duration_slots=1,
                            split_fallback=False
                        ))
                    changed = True
                else:
                    new_lessons.append(L)
            if changed:
                lessons = new_lessons
                domains = {}
                zero_domain_lessons = []
                for L in lessons:
                    dom = self.candidate_assignments(L)
                    domains[L.lesson_id] = dom
                    if not dom:
                        zero_domain_lessons.append(L)

        self.initial_zero_domain = [L.lesson_id for L in zero_domain_lessons]
        if zero_domain_lessons:
            return None, {"feasible": False, "reason": "Lezioni senza candidati", "lessons": self.initial_zero_domain}

        ordered_lessons = self.order_lessons(lessons, domains)
        solution = self.search(ordered_lessons, domains)
        if solution is None:
            return None, {"feasible": False, "reason": "Backtracking fallito (probabile conflitto di risorse).", "zero_domain": self.initial_zero_domain}
        solution.sort(key=lambda a: (a.date, a.start_slot, a.class_or_group_id))
        return solution, {"feasible": True, "zero_domain": self.initial_zero_domain}

# -----------------------------
# Postgres IO
# -----------------------------

def connect_pg(dsn: str):
    if HAVE_PSYCOPG3:
        return psycopg.connect(dsn)  # psycopg v3
    return psycopg.connect(dsn)      # psycopg2 signature

def load_config_from_db(conn, start_date: date, days: Optional[int], weeks: Optional[int]):
    cur = conn.cursor()
    cur.execute("SELECT class_id, name FROM classes ORDER BY class_id")
    classes = {r[0]: Class(r[0], r[1]) for r in cur.fetchall()}

    cur.execute("SELECT subject_id, name, COALESCE(block_len_min,1), COALESCE(macro_preferred,false) FROM subjects ORDER BY subject_id")
    subjects = {r[0]: Subject(r[0], r[1], int(r[2]), bool(r[3])) for r in cur.fetchall()}

    cur.execute("SELECT teacher_id, name FROM teachers ORDER BY teacher_id")
    teachers = {r[0]: Teacher(r[0], r[1]) for r in cur.fetchall()}

    cur.execute("SELECT teacher_id, subject_id FROM teacher_subjects")
    teacher_subjects: Dict[str, Set[str]] = {}
    for tid, sid in cur.fetchall():
        teacher_subjects.setdefault(tid, set()).add(sid)

    cur.execute("SELECT merge_id, name FROM merge_groups ORDER BY merge_id")
    merge_groups: Dict[str, MergeGroup] = {}
    for mid, name in cur.fetchall():
        merge_groups[mid] = MergeGroup(mid, name, members=[])
    cur.execute("SELECT merge_id, class_id FROM merge_members")
    for mid, cid in cur.fetchall():
        if mid in merge_groups:
            merge_groups[mid].members.append(cid)

    # availability rows -> aggregate to set per (teacher, weekday)
    cur.execute("SELECT teacher_id, weekday, slot_label FROM teacher_availability")
    av_map: Dict[Tuple[str,int], Set[str]] = {}
    for tid, wd, slot_label in cur.fetchall():
        av_map.setdefault((tid, int(wd)), set()).add(slot_label)
    availabilities: List[Availability] = [Availability(tid, wd, slots) for (tid, wd), slots in av_map.items()]

    # holidays
    cur.execute("SELECT dt FROM holidays")
    holidays = {r[0] for r in cur.fetchall()}

    # school days: Mon-Fri only, skip holidays
    school_days: List[date] = []
    if weeks is not None and weeks > 0:
        d = start_date
        while d.weekday() != 0:
            d += timedelta(days=1)
        total_days = weeks * 7
        end = d + timedelta(days=total_days-1)
        dd = d
        while dd <= end:
            if dd.weekday() < 5 and dd not in holidays:
                school_days.append(dd)
            dd += timedelta(days=1)
    elif days is not None and days > 0:
        dd = start_date
        while len(school_days) < days:
            if dd.weekday() < 5 and dd not in holidays:
                school_days.append(dd)
            dd += timedelta(days=1)
    else:
        raise ValueError("Specificare --days o --weeks.")

    cur.close()
    return classes, subjects, teachers, teacher_subjects, merge_groups, availabilities, SchoolCalendar(school_days)

def export_csv(schedule: List[Assignment], classes: Dict[str, Class], subjects: Dict[str, Subject], teachers: Dict[str, Teacher], merge_groups: Dict[str, MergeGroup], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Data","Giorno","Inizio","Fine","Classe/Group","Materia","Docente"])
        for a in schedule:
            name = ""
            if a.class_or_group_id.startswith("C:"):
                cid = a.class_or_group_id.split("C:",1)[1]
                name = classes[cid].name
            else:
                gid = a.class_or_group_id.split("G:",1)[1]
                name = merge_groups[gid].name
            w.writerow([a.date.isoformat(), weekday_it(a.date), a.start_slot, a.end_slot, name, subjects[a.subject_id].name, teachers[a.teacher_id].name])

def ics_escape(text: str) -> str:
    return text.replace("\\","\\\\").replace("\n","\\n").replace(",","\\,").replace(";","\\;")

def export_ics_per_class(schedule: List[Assignment], classes: Dict[str, Class], subjects: Dict[str, Subject], teachers: Dict[str, Teacher], merge_groups: Dict[str, MergeGroup], folder: str, tzid: str = "Europe/Rome"):
    os.makedirs(folder, exist_ok=True)
    class_events: Dict[str, List[Assignment]] = {cid: [] for cid in classes.keys()}
    for a in schedule:
        if a.class_or_group_id.startswith("C:"):
            cid = a.class_or_group_id.split("C:",1)[1]
            class_events[cid].append(a)
        else:
            gid = a.class_or_group_id.split("G:",1)[1]
            members = merge_groups[gid].members
            for cid in members:
                class_events[cid].append(a)

    for cid, events in class_events.items():
        path = os.path.join(folder, f"orario_classe_{cid}.ics")
        with open(path, "w", encoding="utf-8") as f:
            f.write("BEGIN:VCALENDAR\r\nVERSION:2.0\r\nPRODID:-//HybridTimetable//IT//\r\n")
            for idx, a in enumerate(events, start=1):
                start_dt = dt_local(a.date, a.start_slot)
                all_slots = [s for s in SLOTS_ALL if s <= "13:00"]
                if a.end_slot in all_slots:
                    i = all_slots.index(a.end_slot)
                    if i+1 < len(all_slots):
                        end_label = all_slots[i+1]
                        end_dt = dt_local(a.date, end_label)
                    else:
                        end_dt = dt_local(a.date, a.end_slot) + timedelta(hours=1)
                else:
                    end_dt = dt_local(a.date, a.end_slot) + timedelta(hours=1)

                def fmt(dt: datetime) -> str:
                    return dt.strftime("%Y%m%dT%H%M%S")

                title = f"{subjects[a.subject_id].name} - {teachers[a.teacher_id].name}"
                f.write("BEGIN:VEVENT\r\n")
                f.write(f"UID:CLASS-{cid}-{a.lesson_id}-{idx}@hybrid\r\n")
                f.write(f"DTSTART;TZID={tzid}:{fmt(start_dt)}\r\n")
                f.write(f"DTEND;TZID={tzid}:{fmt(end_dt)}\r\n")
                f.write(f"SUMMARY:{ics_escape(title)}\r\n")
                f.write("END:VEVENT\r\n")
            f.write("END:VCALENDAR\r\n")

def export_ics_per_teacher(schedule: List[Assignment], teachers: Dict[str, Teacher], subjects: Dict[str, Subject], classes: Dict[str, Class], merge_groups: Dict[str, MergeGroup], folder: str, tzid: str = "Europe/Rome"):
    os.makedirs(folder, exist_ok=True)
    teacher_events: Dict[str, List[Assignment]] = {tid: [] for tid in teachers.keys()}
    for a in schedule:
        teacher_events[a.teacher_id].append(a)

    for tid, events in teacher_events.items():
        path = os.path.join(folder, f"orario_docente_{tid}.ics")
        with open(path, "w", encoding="utf-8") as f:
            f.write("BEGIN:VCALENDAR\r\nVERSION:2.0\r\nPRODID:-//HybridTimetable//IT//\r\n")
            for idx, a in enumerate(events, start=1):
                start_dt = dt_local(a.date, a.start_slot)
                all_slots = [s for s in SLOTS_ALL if s <= "13:00"]
                if a.end_slot in all_slots:
                    i = all_slots.index(a.end_slot)
                    if i+1 < len(all_slots):
                        end_label = all_slots[i+1]
                        end_dt = dt_local(a.date, end_label)
                    else:
                        end_dt = dt_local(a.date, a.end_slot) + timedelta(hours=1)
                else:
                    end_dt = dt_local(a.date, a.end_slot) + timedelta(hours=1)

                def fmt(dt: datetime) -> str:
                    return dt.strftime("%Y%m%dT%H%M%S")
                if a.class_or_group_id.startswith("C:"):
                    cid = a.class_or_group_id.split("C:",1)[1]
                    entity = f"Classe {classes[cid].name}"
                else:
                    gid = a.class_or_group_id.split("G:",1)[1]
                    entity = f"Gruppo {merge_groups[gid].name}"
                title = f"{subjects[a.subject_id].name} - {entity}"
                f.write("BEGIN:VEVENT\r\n")
                f.write(f"UID:TEACHER-{tid}-{a.lesson_id}-{idx}@hybrid\r\n")
                f.write(f"DTSTART;TZID={tzid}:{fmt(start_dt)}\r\n")
                f.write(f"DTEND;TZID={tzid}:{fmt(end_dt)}\r\n")
                f.write(f"SUMMARY:{ics_escape(title)}\r\n")
                f.write("END:VEVENT\r\n")
            f.write("END:VCALENDAR\r\n")

def write_schedule_to_db(conn, schedule: List[Assignment], clear_range: bool = False):
    if not schedule:
        return
    dmin = min(a.date for a in schedule)
    dmax = max(a.date for a in schedule)
    cur = conn.cursor()
    if clear_range:
        cur.execute("DELETE FROM schedule WHERE date BETWEEN %s AND %s", (dmin, dmax))
    for a in schedule:
        if a.class_or_group_id.startswith("C:"):
            class_or_group_type = "C"
            cid = a.class_or_group_id.split("C:",1)[1]
            gid = None
        else:
            class_or_group_type = "G"
            gid = a.class_or_group_id.split("G:",1)[1]
            cid = None
        cur.execute("""
            INSERT INTO schedule (date, start_slot, end_slot, class_or_group_type, class_id, merge_id, subject_id, teacher_id)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """, (a.date, a.start_slot, a.end_slot, class_or_group_type, cid, gid, a.subject_id, a.teacher_id))
    conn.commit()
    cur.close()

# -----------------------------
# Main
# -----------------------------

def _plans_from_db(conn) -> List[ClassSubjectPlan]:
    cur = conn.cursor()
    cur.execute("""
        SELECT plan_id, class_id, merge_id, subject_id, total_hours
        FROM class_subject_plan
        ORDER BY plan_id
    """)
    plans: List[ClassSubjectPlan] = []
    for plan_id, class_id, merge_id, subject_id, total_hours in cur.fetchall():
        if class_id and not merge_id:
            cid = f"C:{class_id}"
        elif merge_id and not class_id:
            cid = f"G:{merge_id}"
        else:
            raise ValueError(f"Piano {plan_id}: specificare solo class_id O merge_id.")
        plans.append(ClassSubjectPlan(str(plan_id), cid, subject_id, int(total_hours)))
    cur.close()
    return plans

def main():
    ap = argparse.ArgumentParser(description="Hybrid School Timetable (Postgres)")
    ap.add_argument("--dsn", required=True, help="Postgres DSN, es: postgresql://user:pass@localhost:5432/scuola_orari")
    ap.add_argument("--start", type=lambda s: date.fromisoformat(s), help="Data di inizio (YYYY-MM-DD)")
    ap.add_argument("--days", type=int, help="Numero di giorni scolastici da generare (Mon-Fri, salta weekend/festivi)")
    ap.add_argument("--weeks", type=int, help="Numero di settimane Mon-Fri da generare (ignora --days)")
    ap.add_argument("--out", default="./timetable_output", help="Cartella di output per CSV/ICS")
    ap.add_argument("--use-afternoon", action="store_true", help="Includi 14:30 e 15:30")
    ap.add_argument("--write-db", action="store_true", help="Scrivi la soluzione nella tabella schedule")
    ap.add_argument("--clear-range", action="store_true", help="Cancella righe esistenti in schedule nell'intervallo di date della soluzione")
    args = ap.parse_args()

    if args.start is None:
        today = date.today()
        args.start = today + timedelta(days=((7 - today.weekday()) % 7))
    if args.days is None and args.weeks is None:
        args.weeks = 1

    os.makedirs(args.out, exist_ok=True)

    with connect_pg(args.dsn) as conn:
        classes, subjects, teachers, teacher_subjects, merge_groups, availabilities, school_calendar = load_config_from_db(
            conn, start_date=args.start, days=args.days, weeks=args.weeks
        )
        solver = TimetableSolver(
            classes=classes,
            subjects=subjects,
            teachers=teachers,
            teacher_subjects=teacher_subjects,
            availability=availabilities,
            class_plans=_plans_from_db(conn),
            merge_groups=merge_groups,
            school_calendar=school_calendar,
            use_afternoon=args.use_afternoon,
        )
        schedule, meta = solver.solve(allow_macro_split_fallback=True, try_split_on_fail=True)
        if not meta.get("feasible"):
            print("⚠️ Nessuna soluzione trovata.")
            print(meta)
            sys.exit(2)

        csv_path = os.path.join(args.out, "schedule.csv")
        export_csv(schedule, classes, subjects, teachers, merge_groups, csv_path)
        export_ics_per_class(schedule, classes, subjects, teachers, merge_groups, os.path.join(args.out, "ics_classi"))
        export_ics_per_teacher(schedule, teachers, subjects, classes, merge_groups, os.path.join(args.out, "ics_docenti"))
        print(f"OK. CSV: {csv_path}")

        if args.write_db:
            write_schedule_to_db(conn, schedule, clear_range=args.clear_range)
            print("Soluzione scritta su DB (tabella 'schedule').")

if __name__ == "__main__":
    main()
