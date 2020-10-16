package models

type Job struct {
	Val float64 `json:"id"`
	Key string  `json:"name"`
}

type Entry struct {
	Val float64
	Key int
}

type Entries []Entry

func (s Entries) Len() int           { return len(s) }
func (s Entries) Less(i, j int) bool { return s[i].val < s[j].val }
func (s Entries) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
